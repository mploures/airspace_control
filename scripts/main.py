#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vant_ros.py — ROS + UltraDES com isolamento do runtime .NET

Comportamento:
- Se DOTNET_ROOT detectado: tenta CoreCLR.
- Senão: cai para Mono, mas com spawn e sem carregar UltraDES no processo pai.

Ambos os casos evitam o crash do Mono (jit_tls) por evitar fork após carregar CLR.
"""
import math
import os
import sys
import time
import argparse
from multiprocessing import Process, set_start_method
import re
from std_msgs.msg import String 
import rospy
import rospkg

def _cb_event_with_move(inst, vant):
    """
    Cria e retorna a função de callback que processa eventos do /event,
    integrando a transição do supervisor (inst) com a definição do objetivo
    físico (vant.goal), a lógica de parada e a geração automática de 'libera_*'.
    """
    from std_msgs.msg import String

    def callback(msg: String):
        ev = str(msg.data or "").strip()
        vant.ros_node.loginfo(f"[{vant.name}] ➡️ Recebido evento: '{ev}'")

        # 1. Lógica de PING
        if ev == "ping":
            inst._publish_ros()
            return

        # 2. Lógica de Transição de Estado do Supervisor
        if not inst.step(ev):
            # Evento não é deste VANT (id diferente, etc.)
            return

        ev_gen = inst.to_generic(ev)
        vant.ros_node.loginfo(f"[{vant.name}] 🔄 Transição OK: '{ev}' (gen='{ev_gen}')")

        # 3. Lógica de Decisão de Movimento/Parada
        if ev_gen.startswith("pega_"):
            # É um evento de movimento -> Define o novo objetivo
            pos_entry = inst.posicoes.get(ev_gen)

            # Esperamos: pos_entry == (event_obj, (label, (x, y)))
            if pos_entry is not None and isinstance(pos_entry, tuple) and len(pos_entry) == 2:
                event_obj, coord_entry = pos_entry

                if isinstance(coord_entry, tuple) and len(coord_entry) == 2:
                    label, coordinates = coord_entry
                    original_x, original_y = coordinates

                    #print(f"[DEBUG] pos_entry: {pos_entry}")
                    #print(f"[DEBUG] event_obj: {event_obj}")
                    print(f"[DEBUG] label: {label}, x: {original_x}, y: {original_y}")

                    if not isinstance(original_x, (int, float)) or not isinstance(original_y, (int, float)):
                        vant.ros_node.logerr(
                            f"[{vant.name}] ERRO: Coordenadas obtidas para '{ev_gen}' não são números: {pos_entry}"
                        )
                        # não queremos liberar aresta errada depois
                        vant._pending_release_event = None
                        vant._stop_movement()
                        return

                    # Define o objetivo físico (coordenadas no Stage)
                    vant.goal = (original_x, original_y)
                    vant.ros_node.loginfo(
                        f"[{vant.name}] 🎯 Meta Supervisor. Destino REAL (Stage): ({original_x:.2f}, {original_y:.2f})."
                    )

                    # ---- Cálculo do evento de liberação correspondente ----
                    # ev_gen = "pega_<origem><destino>"
                    libera_gen = "libera_" + ev_gen[len("pega_"):]
                    # Usa o mapeamento genérico -> evento com sufixo _{id}
                    libera_id = inst.event_map.get(libera_gen)

                    if libera_id is None:
                        # Se por algum motivo não existir no supervisor
                        vant._pending_release_event = None
                        vant.ros_node.logwarn(
                            f"[{vant.name}] Não encontrei evento de liberação correspondente para '{ev_gen}' "
                            f"(esperado gen='{libera_gen}')."
                        )
                    else:
                        # Guardamos no VANT para publicar quando o objetivo for atingido
                        vant._pending_release_event = libera_id
                        vant.ros_node.loginfo(
                            f"[{vant.name}] ⏱️ Ao atingir o objetivo será publicado '{libera_id}' em /event."
                        )

                    # Inicia o loop de controle até chegar no destino
                    vant.spin()
                    return

                else:
                    vant._pending_release_event = None
                    vant._stop_movement()
                    vant.ros_node.logwarn(
                        f"[{vant.name}] Formato inválido de coordenadas internas para '{ev_gen}': {coord_entry}"
                    )
            else:
                vant._pending_release_event = None
                vant._stop_movement()
                vant.ros_node.logwarn(
                    f"[{vant.name}] Transição 'pega_' ocorreu, mas coordenada para '{ev_gen}' "
                    f"não encontrada ou inválida: {pos_entry}"
                )

        elif ev_gen.startswith("libera_"):
            # Evento de liberação vindo de fora (painel, etc.)
            vant._pending_release_event = None
            vant._stop_movement()
            vant.ros_node.loginfo(
                f"[{vant.name}] 🛑 Parada forçada por evento de liberação recebido ('{ev_gen}')."
            )

        else:
            # Evento de controle (não-movimento)
            vant._pending_release_event = None
            vant._stop_movement()
            vant.ros_node.loginfo(
                f"[{vant.name}] 🛑 Evento não-movimento ('{ev_gen}'). Meta física resetada."
            )

    return callback


def resolve_grafo_path(rel="graph/sistema_logistico/grafo_recortado.txt"):
    try:
        rp = rospkg.RosPack()
        base = rp.get_path("airspace_control")
        p = os.path.join(base, rel)
        if os.path.isfile(p):
            print(f"[INFO] Arquivo encontrado: {p}")
            return p
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    base2 = os.path.abspath(os.path.join(here, ".."))
    p2 = os.path.join(base2, rel)
    if os.path.isfile(p2):
        print(f"[INFO] Arquivo encontrado: {p2}")
        return p2
    p3 = os.path.join(os.path.expanduser("~/catkin_ws/src/airspace_control"), rel)
    if os.path.isfile(p3):
        print(f"[INFO] Arquivo encontrado: {p3}")
        return p3
    print("[ERRO] Não foi possível localizar o grafo.")
    return os.path.join(base2, rel)


def run_vant_instance(vant_id: int, grafo_path: str, init_node: str, backend: str):
    """
    Processo filho: configura pythonnet, então importa UltraDES/controle e roda.
    backend ∈ {"coreclr","mono"}
    """
    try:
        # 1) Ambiente pythonnet ANTES de qualquer import do ultrades
        os.environ.setdefault("PYTHONNET_CLEANUP", "0")
        os.environ.setdefault("DOTNET_NOLOGO", "1")

        if backend == "coreclr":
            os.environ["PYTHONNET_RUNTIME"] = "coreclr"
        else:
            os.environ["PYTHONNET_RUNTIME"] = "mono"
            # Flags do Mono para estabilidade quando threads ROS entram:
            os.environ.setdefault("MONO_THREADS_SUSPEND", "preemptive")
            os.environ.setdefault("MONO_NO_SMP", "1")  # opcional em CPUs antigas/VMs

        # 2) Tenta carregar explicitamente o runtime
        try:
            from pythonnet import load as _pyload
            _pyload(backend)
            print(f"[VANT {vant_id}] pythonnet carregado com backend={backend}")
        except Exception as e:
            print(f"[WARN] Falha ao carregar backend={backend}: {e}")
            if backend == "coreclr":
                print("[WARN] Caindo para backend=mono.")
                os.environ["PYTHONNET_RUNTIME"] = "mono"
                from pythonnet import load as _pyload2
                _pyload2("mono")

        # 3) IMPORTS (só no filho)
        from airspace_core.controlador_vant import GenericVANTModel, VANTInstance, VANT

        print(f"[VANT {vant_id}] Iniciando processo...")
        if not os.path.isfile(grafo_path):
            print(f"[ERRO VANT {vant_id}] Arquivo do grafo não encontrado: {grafo_path}")
            return

        print(f"[VANT {vant_id}] Construindo modelo...")
        model = GenericVANTModel(grafo_txt=grafo_path, init_node=init_node)

        print(f"[VANT {vant_id}] Computando supervisor monolítico (GEN)...")
        S = model.compute_monolithic_supervisor()

    
        print(f"[VANT {vant_id}] Criando VANTInstance ROS...")
        inst = VANTInstance(
            model=model,
            id_num=vant_id,
            supervisor_mono=S,
            obj_vant=None,
            enable_ros=True,
            node_name=f"supervisor_vant_{vant_id}"
        )

        vant_fisico = VANT(f"vant_{vant_id}", rospy)

        callback_final = _cb_event_with_move(inst, vant_fisico)

        inst.sub_event.unregister() 

        inst.sub_event = rospy.Subscriber("/event", String, callback_final, queue_size=50)

        print(f"[VANT {vant_id}] Callback de evento atualizado com lógica de movimento.")
        print(f"[VANT {vant_id}] Rodando ROS spin...")
        inst.run()

    except Exception as e:
        print(f"[ERRO VANT {vant_id}] {e}")
        import traceback; traceback.print_exc()

def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--nvant", type=int, default=1, help="Número de VANTs a serem criados")
    parser.add_argument("--grafo", default=resolve_grafo_path(), help="Caminho para grafo")
    parser.add_argument("--init", default="VERTIPORT_0", help="Nó inicial")
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.grafo):
        print(f"[ERRO] Arquivo do grafo não encontrado: {args.grafo}")
        return 1

    # Gera lista de IDs de 0 a nvant-1
    ids = list(range(args.nvant))

    backend = "mono"
    print(f"[INFO] Backend preferido: {backend}")

    print(f"[INFO] Iniciando {len(ids)} VANT(s): {ids}")
    print(f"[INFO] Grafo: {args.grafo}")
    print(f"[INFO] Nó inicial: {args.init}")
    print("[INFO] Garanta que 'roscore' está ativo e abra o control_panel em outro terminal.")
    print("-" * 60)

    procs = []
    try:
        for vid in ids:
            p = Process(target=run_vant_instance, args=(vid, args.grafo, args.init, backend), daemon=True)
            p.start()
            procs.append(p)
            time.sleep(2.0)

        print(f"[INFO] {len(procs)} processo(s) iniciado(s). Ctrl+C para encerrar.")
        for p in procs:
            p.join()

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=2)
        print("[INFO] Finalizado.")
    return 0

if __name__ == "__main__":
    sys.exit(main())