#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_vant_ros.py — ROS + UltraDES com isolamento do runtime .NET

Comportamento:
- Se DOTNET_ROOT detectado: tenta CoreCLR.
- Senão: cai para Mono, mas com spawn e sem carregar UltraDES no processo pai.

Ambos os casos evitam o crash do Mono (jit_tls) por evitar fork após carregar CLR.
"""

import os
import sys
import time
import argparse
from multiprocessing import Process, set_start_method

import rospy
import rospkg



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
        from airspace_core.controlador_vant import GenericVANTModel, VANTInstance

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

        print(f"[VANT {vant_id}] Rodando ROS spin...")
        inst.run()

    except Exception as e:
        print(f"[ERRO VANT {vant_id}] {e}")
        import traceback; traceback.print_exc()

def main():
    # Método 'spawn' impede herdar CLR já carregado
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="0", help="IDs dos VANTs separados por vírgula (ex.: 0,1)")
    parser.add_argument("--grafo", default=resolve_grafo_path(), help="Caminho para grafo")
    parser.add_argument("--init", default="VERTIPORT_0", help="Nó inicial")
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.grafo):
        print(f"[ERRO] Arquivo do grafo não encontrado: {args.grafo}")
        return 1

    try:
        ids = [int(x.strip()) for x in args.ids.split(",") if x.strip() != ""]
    except ValueError:
        print("[ERRO] --ids precisa conter inteiros separados por vírgula.")
        return 1

    backend = "mono"
    print(f"[INFO] Backend preferido: {backend}")
    if backend == "mono":
        print("[INFO] DOTNET_ROOT não encontrado — usando Mono com spawn e import tardio.")

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
