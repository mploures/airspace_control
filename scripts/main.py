#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import subprocess
import signal
import rospy
import networkx as nx

from airspace_control.srv import GetBattery, GotoXY  # noqa: F401
from airspace_core.controlador_vant import ControladorVANT

# ------------------------------ util ------------------------------
def carregar_grafo_txt(caminho_arquivo):
    G = nx.Graph()
    pos = {}
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        linhas = [l.strip() for l in f.readlines() if l.strip()]
    for linha in linhas[1:]:
        m = re.match(r'^([^,]+),([^,]+),\(([^,]+),([^)]+)\),(.*)$', linha)
        if not m:
            rospy.logwarn(f"[grafo.txt] Linha ignorada (formato incorreto): {linha}")
            continue
        tipo_no = m.group(1).strip()
        label   = m.group(2).strip()
        x = float(m.group(3).strip()); y = float(m.group(4).strip())
        pos[label] = (x, y)
        G.add_node(label, tipo=tipo_no)
        resto = m.group(5).strip()
        if resto:
            conectados = [p.strip() for p in resto.split(',') if p.strip()]
            for c in conectados:
                if c != label:
                    G.add_edge(label, c)
    return G, pos

class DummyVANT:
    def __init__(self, name): self.name = name

# ------------------------------ WORKER ------------------------------
def run_worker(role, graph_path, start_node, robot_name, spawn_edges=False):
    # NUNCA chame init_node aqui. Quem fará é a AutomatonNode.
    G_raw, pos = carregar_grafo_txt(graph_path)
    G = nx.MultiGraph()
    G.add_nodes_from(G_raw.nodes(data=True))
    G.add_edges_from(G_raw.edges())
    for n in G.nodes():
        if n in pos: G.nodes[n]["pos"] = tuple(pos[n])
    if start_node not in G.nodes: start_node = next(iter(G.nodes))

    ROLE_TO_METHOD = {
        "movimento":    "_dfa_movimento_as_node",
        "modos":        "_dfa_modos_as_node",
        "mapa":         "_dfa_mapa_as_node",
        "suporte":      "_dfas_suporte_as_nodes",
        "bateria_mov":  "_dfa_bateria_mov_as_node",
        "arestas":      "_dfas_arestas_as_nodes",
    }
    if role not in ROLE_TO_METHOD:
        raise ValueError(f"Role inválido: {role}")

    original_builder = ControladorVANT._construir_automatos

    def only_one_automaton(self):
        method_name = ROLE_TO_METHOD[role]
        if method_name == "_dfas_arestas_as_nodes" and not spawn_edges:
            rospy.logwarn("[worker] Arestas desabilitadas (spawn_edges=False).")
            return
        getattr(self, method_name)()

    ControladorVANT._construir_automatos = only_one_automaton

    vant = DummyVANT(name=robot_name)
    _ctrl = ControladorVANT(id_vant=robot_name, obj_vant=vant, grafo=G,
                            porto_inicial=start_node, pos_dict=pos)

    rospy.loginfo(f"[worker:{role}] pronto para {robot_name}.")
    rospy.spin()

    ControladorVANT._construir_automatos = original_builder

# ------------------------------ COORDENADOR ------------------------------
def run_coordinator(graph_path, start_node, robot_name, spawn_edges=False):
    # Aqui podemos inicializar, pois este processo NÃO cria AutomatonNode.
    rospy.init_node("uav_system", anonymous=False)

    G_raw, _ = carregar_grafo_txt(graph_path)
    rospy.loginfo(f"[main] Grafo: {G_raw.number_of_nodes()} nós, {G_raw.number_of_edges()} arestas")
    rospy.loginfo(f"[main] Robot: {robot_name} | Nó inicial: {start_node}")

    roles = ["movimento", "modos", "mapa", "suporte", "bateria_mov"]
    if spawn_edges:
        roles.append("arestas")  # cuidado: MUITOS processos

    this_script = os.path.abspath(__file__)
    py = sys.executable
    children = []

    def spawn(role):
        args = [
            py, this_script,
            "--role", role,
            "--graph_path", graph_path,
            "--start_node", start_node,
            "--robot_name", robot_name,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(args, env=env)
        rospy.loginfo(f"[main] Spawned worker '{role}' (pid={p.pid}) para {robot_name}.")
        return p

    for r in roles:
        children.append(spawn(r))

    def _shutdown(_sig=None, _frm=None):
        rospy.loginfo("[main] Encerrando workers…")
        for p in children:
            try: p.send_signal(signal.SIGINT)
            except Exception: pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    rospy.loginfo("[main] Sistema pronto. Publique eventos no tópico /event (std_msgs/String).")
    rospy.spin()
    _shutdown()

# ------------------------------ ARGS ------------------------------
def parse_args():
    # IMPORTANTÍSSIMO: remove __name:=..., __log:=..., remappings etc.
    argv = rospy.myargv(argv=sys.argv)
    ap = argparse.ArgumentParser(add_help=False)  # não capturar -h do ROS
    ap.add_argument("--role", default=None)
    ap.add_argument("--graph_path", default=None)
    ap.add_argument("--start_node", default=None)
    ap.add_argument("--robot_name", default=None)
    ap.add_argument("--spawn_edges", action="store_true")
    args, _unknown = ap.parse_known_args(argv[1:])  # ignora o resto
    return args

# ------------------------------ MAIN ------------------------------
def main():
    args = parse_args()

    # WORKER (chamado pelos subprocessos)
    if args.role:
        if not (args.graph_path and args.start_node and args.robot_name):
            print("[worker] Faltam --graph_path/--start_node/--robot_name.", file=sys.stderr)
            sys.exit(2)
        run_worker(role=args.role,
                   graph_path=args.graph_path,
                   start_node=args.start_node,
                   robot_name=args.robot_name,
                   spawn_edges=args.spawn_edges)
        return

    # COORDENADOR (chamado pelo roslaunch)
    # Aqui pegamos os params privados do node lançados no .launch
    rospy.init_node("uav_system_param", anonymous=True)
    graph_path = rospy.get_param("~graph_path", "graph/sistema_logistico/grafo.txt")
    start_node = rospy.get_param("~start_node", "VANTPORT_0")
    robot_name = rospy.get_param("~robot_name", "vant_0")
    spawn_edges = rospy.get_param("~spawn_edges", False)
    rospy.signal_shutdown("param-ok")

    run_coordinator(graph_path, start_node, robot_name, spawn_edges=spawn_edges)

if __name__ == "__main__":
    main()
