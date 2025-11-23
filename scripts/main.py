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
    integrando:
      - transição do supervisor (inst)
      - definição do objetivo físico (vant.goal)
      - lógica de parada
      - geração automática de 'libera_*'
      - agendamento de eventos NÃO-CONTROLÁVEIS de término de serviço
        (fim_trabalho_*, fim_carregar_*)
      - restauração da bateria física no fim do carregamento.
    """
    from std_msgs.msg import String
    import rospy
    import threading

    def _schedule_uncontrollable_response(ev_id: str, ev_gen: str):
        """
        Agenda, após um tempo, o evento NÃO-CONTROLÁVEL de resposta:

        - comeca_trabalho_X  -> fim_trabalho_X
        - carregar_X         -> fim_carregar_X

        Sempre publica com o MESMO id do VANT (sufixo _{id}).
        """
        # Descobre qual é o "fim_*" correspondente
        if ev_gen.startswith("comeca_trabalho_"):
            alvo = ev_gen[len("comeca_trabalho_"):]          # FORNECEDOR_0, CLIENTE_0, ...
            fim_gen = f"fim_trabalho_{alvo}"
        elif ev_gen.startswith("carregar_"):
            alvo = ev_gen[len("carregar_"):]                 # ESTACAO_0, ...
            fim_gen = f"fim_carregar_{alvo}"
        else:
            # Não é um evento de início de serviço conhecido
            return

        # Usa o mapeamento genérico -> com id do VANT
        fim_id = inst.event_map.get(fim_gen)
        if fim_id is None:
            vant.ros_node.logwarn(
                f"[{vant.name}] Não encontrei evento não-controlável de término para '{ev_gen}' "
                f"(esperado gen='{fim_gen}')."
            )
            return

        # Tempo de serviço (pode vir de parâmetro do modelo)
        duracao = getattr(inst.model, "tempo_servico_padrao", 5.0)  # segundos

        vant.ros_node.loginfo(
            f"[{vant.name}] ⏱️ Agendando evento não-controlável '{fim_id}' para daqui a {duracao:.1f} s "
            f"(resposta a '{ev_id}')."
        )

        def _fire():
            try:
                # Tenta usar um publisher já existente no VANTInstance, se tiver
                if hasattr(inst, "pub_cmd_event"):
                    pub = inst.pub_cmd_event
                elif hasattr(vant, "pub_event_out"):
                    pub = vant.pub_event_out
                else:
                    # fallback simples
                    pub = rospy.Publisher("/event", String, queue_size=10)

                vant.ros_node.loginfo(
                    f"[{vant.name}] ⏱️ Timer disparou. Publicando evento não-controlável '{fim_id}' em /event."
                )
                pub.publish(String(data=fim_id))
            except Exception as e:
                vant.ros_node.logerr(
                    f"[{vant.name}] Erro ao publicar evento temporizado '{fim_id}': {e}"
                )

        t = threading.Timer(duracao, _fire)
        t.daemon = True
        t.start()

    def callback(msg: String):
        ev = str(msg.data or "").strip()
        vant.ros_node.loginfo(f"[{vant.name}] ➡️ Recebido evento: '{ev}'")

        # 1. Lógica de PING
        if ev == "ping":
            inst._publish_ros()
            return

        # 2. Lógica de Transição de Estado do Supervisor
        if not inst.step(ev):
            # Evento não é deste VANT (id diferente, etc.) ou foi bloqueado pelo UTM
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

        elif ev_gen.startswith("comeca_trabalho_") or ev_gen.startswith("carregar_"):
            # Início de serviço local (trabalho ou carregamento):
            # - Reseta movimento físico
            # - Agenda o evento NÃO-CONTROLÁVEL de término correspondente
            vant._pending_release_event = None
            vant._stop_movement()
            vant.ros_node.loginfo(
                f"[{vant.name}] 🛠️ Início de serviço '{ev_gen}'. Agendando término não-controlável."
            )
            _schedule_uncontrollable_response(ev, ev_gen)

        elif ev_gen.startswith("fim_carregar_"):
            # Fim de carregamento:
            # - Para movimento (por segurança)
            # - Restaura bateria física do VANT para 100%
            vant._pending_release_event = None
            vant._stop_movement()
            try:
                if hasattr(vant, "restore_full_battery"):
                    vant.restore_full_battery()
                else:
                    # fallback se por algum motivo o método não existir
                    vant.soc = 1.0
                    if hasattr(vant, "_low_batt_sent"):
                        vant._low_batt_sent = False
                    vant.last_batt_ts = time.time()

                vant.ros_node.loginfo(
                    f"[{vant.name}] 🔋 Evento '{ev_gen}' recebido — bateria recarregada para carga máxima."
                )
            except Exception as e:
                vant.ros_node.logerr(
                    f"[{vant.name}] Erro ao processar recarga de bateria em '{ev_gen}': {e}"
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

def run_utm_node(num_agents: int, grafo_path: str, init_node: str, backend: str):
    """
    Processo principal: configura pythonnet, então importa UltraDES/controle e roda o nó UTM central.
    backend ∈ {"coreclr","mono"}
    
    Args:
        num_agents (int): Número de agentes (VANTs) a serem rastreados.
        grafo_path (str): Caminho para o arquivo de grafo.
        init_node (str): Nome do nó inicial no grafo.
        backend (str): Backend do pythonnet ('coreclr' ou 'mono').
    """
    try:
        from airspace_core.UTM import UTMROSInterface

        # 1) Configuração do ambiente pythonnet ANTES de qualquer import do UltraDES
        # Garante que os imports de DES funcionem corretamente em um processo separado.
        os.environ.setdefault("PYTHONNET_CLEANUP", "0")
        os.environ.setdefault("DOTNET_NOLOGO", "1")

        target_backend = backend.lower()
        if target_backend not in ["coreclr", "mono"]:
            print(f"[UTM] Backend inválido '{backend}'. Usando 'coreclr' por padrão.")
            target_backend = "coreclr"
            
        os.environ["PYTHONNET_RUNTIME"] = target_backend

        # 2) Tenta carregar explicitamente o runtime
        try:
            from pythonnet import load as _pyload
            _pyload(target_backend)
            print(f"[UTM] pythonnet carregado com backend={target_backend}")
        except Exception as e:
            print(f"[WARN UTM] Falha ao carregar backend={target_backend}: {e}")
            if target_backend == "coreclr":
                print("[WARN UTM] Caindo para backend=mono.")
                os.environ["PYTHONNET_RUNTIME"] = "mono"
                from pythonnet import load as _pyload2
                _pyload2("mono")
        
        # 3) IMPORTS (após a configuração do runtime)
        # Assumindo que UTMROSInterface, UTMModel, rospy e String estão disponíveis globalmente
        # ou importados neste escopo.

        print(f"[UTM] Iniciando processo de supervisão central...")
        if not os.path.isfile(grafo_path):
            print(f"[ERRO UTM] Arquivo do grafo não encontrado: {grafo_path}")
            return

        print(f"[UTM] Criando UTMROSInterface (Construindo Modelo e Supervisor Monolítico)...")
        
        # A UTMROSInterface faz todo o trabalho de inicialização e computação
        utm_interface = UTMROSInterface(
            grafo_txt=grafo_path,
            init_node=init_node,
            num_agent=num_agents,
            node_name="utm_supervisor_node"
        )
        
        # O callback ROS para /event já está configurado no __init__ da UTMROSInterface
        
        print(f"[UTM] Supervisão Central rodando. Agentes rastreados: {num_agents}.")
        utm_interface.run()

    except Exception as e:
        print(f"[ERRO UTM] Falha fatal no nó UTM: {e}")
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
        p1= Process(target=run_utm_node, args=(args.nvant,args.grafo,args.init,backend), daemon=True)
        p1.start()
        procs.append(p1)
        time.sleep(2.0)

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