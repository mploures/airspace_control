#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import rospy
import networkx as nx

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from airspace_control.srv import GotoXY

from ultrades.automata import *
from ultrades_lib.core.AutomatonNode import AutomatonNode

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import matplotlib.pyplot as plt
import networkx as nx
from ultrades import *


# =============================================================================
# Classe ControladorVANT
# =============================================================================
class ControladorVANT:
    """
    Responsável por associar um VANT físico (objeto da classe VANT)
    aos seus autômatos discretos (movimento, modos, comunicação, bateria, etc.)
    no contexto SEDS / Supervisory Control Theory.
    """

    def __init__(self, id_vant: str, obj_vant, grafo: nx.MultiGraph):
        """
        Parâmetros:
            id_vant : Identificador do VANT no sistema (ex: "vant_0")
            obj_vant : Instância física (contínua) do VANT
            grafo : Grafo logístico do sistema (MultiGraph)
        """
        self.id = id_vant
        self.vant = obj_vant
        self.G = grafo

        # Gera e armazena todos os eventos do sistema
        self.eventos = self.gerarAlfabeto()

        # Inicializa o dicionário de arestas
        self.dict_aresta_eventos = {}

        # Armazena os DFAs construídos
        self.automatos_arestas = []
        self.Uavmodels = []

    # -------------------------------------------------------------------------
    def gerarAlfabeto(self):
        """
        Retorna um dicionário {str(evento): evento} com todos os eventos usados
        no sistema, nomeados com sufixo do VANT (ex: pega_D0L0_vant0).
        """
        G = self.G
        id = self.id
        eventos = {}

        # === 1) Eventos de arestas ===
        for u, v, k, data in G.edges(keys=True, data=True):
            for nome in [
                f"pega_{u}{v}_{id}", f"pega_{v}{u}_{id}",
                f"libera_{u}{v}_{id}", f"libera_{v}{u}_{id}"
            ]:
                eventos[nome] = event(nome, controllable=not nome.startswith("libera"))

        # === 2) Trabalho (fornecedores/clientes) ===
        for v in G.nodes():
            tipo = G.nodes[v].get("tipo", "")
            if tipo in {"fornecedor", "cliente"}:
                nome_ini = f"comeca_trabalho_{v}_{id}"
                nome_fim = f"fim_trabalho_{v}_{id}"
                eventos[nome_ini] = event(nome_ini, controllable=True)
                eventos[nome_fim] = event(nome_fim, controllable=False)

        # === 3) Carregamento (bases) ===
        for v in G.nodes():
            if G.nodes[v].get("tipo", "") == "base":
                nome_ini = f"carregar_{v}_{id}"
                nome_fim = f"fim_carregar_{v}_{id}"
                eventos[nome_ini] = event(nome_ini, controllable=True)
                eventos[nome_fim] = event(nome_fim, controllable=False)

        # === 4) Globais ===
        globais = [
            (f"aceita_tarefa_{id}", True),
            (f"rejeita_tarefa_{id}", True),
            (f"termina_tarefa_{id}", True),
            (f"check_vivacidade_{id}", True),
            (f"bateria_baixa_{id}", False),
        ]
        for nome, ctrl in globais:
            eventos[nome] = event(nome, controllable=ctrl)

        return eventos

    # -------------------------------------------------------------------------
    def construir_automatos(self):
        """Constrói todos os autômatos do VANT com base no grafo e eventos."""
        self._automato_movimento()
        self._automatos_arestas()
        self._automato_modos()
        self._modelos_suporte()
        self._automato_mapa()
        self._automato_bateria_movimento()

    # -------------------------------------------------------------------------
    def _getE(self, nome):
        """Atalho para buscar evento já criado."""
        return self.eventos[nome]

    # -------------------------------------------------------------------------
    def _automato_movimento(self):
        G = self.G
        id = self.id
        self.dict_aresta_eventos = {}

        Parado = state("Parado", marked=True)
        Movendo = state("Movendo")
        trs = []

        for u, v, k, data in G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            pega_uv = self._getE(f"pega_{u}{v}_{id}")
            pega_vu = self._getE(f"pega_{v}{u}_{id}")
            libera_uv = self._getE(f"libera_{u}{v}_{id}")
            libera_vu = self._getE(f"libera_{v}{u}_{id}")
            self.dict_aresta_eventos[chave] = (pega_uv, pega_vu, libera_uv, libera_vu)
            trs += [
                (Parado, pega_uv, Movendo), (Movendo, libera_uv, Parado),
                (Parado, pega_vu, Movendo), (Movendo, libera_vu, Parado)
            ]

        self.A_movimento = dfa(trs, Parado, f"automato_movimento_{id}")
        show_automaton(self.A_movimento)

    # -------------------------------------------------------------------------
    def _automatos_arestas(self):
        G = self.G
        id = self.id

        for u, v, k, data in G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            pega_uv, pega_vu, libera_uv, libera_vu = self.dict_aresta_eventos[chave]
            livre1, oc1 = state(f"livre_{u}{v}", marked=True), state(f"ocupado_{u}{v}")
            livre2, oc2 = state(f"livre_{v}{u}", marked=True), state(f"ocupado_{v}{u}")
            trs1 = [(livre1, pega_uv, oc1), (oc1, libera_uv, livre1)]
            trs2 = [(livre2, pega_vu, oc2), (oc2, libera_vu, livre2)]
            self.automatos_arestas += [
                dfa(trs1, livre1, f"automato_{u}{v}_{id}"),
                dfa(trs2, livre2, f"automato_{v}{u}_{id}")
            ]
        show_automaton(self.automatos_arestas[0])

    # -------------------------------------------------------------------------
    def _automato_modos(self):
        G = self.G
        id = self.id
        geral = state("geral", marked=True)
        trs_modos = []

        # Trabalho
        for v in G.nodes():
            tipo = G.nodes[v].get("tipo", "")
            if tipo in {"fornecedor", "cliente"}:
                evt_ini = self._getE(f"comeca_trabalho_{v}_{id}")
                evt_fim = self._getE(f"fim_trabalho_{v}_{id}")
                s_trab = state(f"trabalhando_{v}")
                trs_modos += [(geral, evt_ini, s_trab), (s_trab, evt_fim, geral)]

        # Carregamento
        for v in G.nodes():
            if G.nodes[v].get("tipo", "") == "base":
                evt_ini = self._getE(f"carregar_{v}_{id}")
                evt_fim = self._getE(f"fim_carregar_{v}_{id}")
                s_carga = state(f"carregando_{v}")
                trs_modos += [(geral, evt_ini, s_carga), (s_carga, evt_fim, geral)]

        A_modos = dfa(trs_modos, geral, f"automato_modos_{id}")
        self.Uavmodels.append(A_modos)
        show_automaton(A_modos)

    # -------------------------------------------------------------------------
    def _modelos_suporte(self):
        id = self.id

        s_com = state("comunicacao_ok", marked=True)
        ace = self._getE(f"aceita_tarefa_{id}")
        rej = self._getE(f"rejeita_tarefa_{id}")
        fim = self._getE(f"termina_tarefa_{id}")
        A_com = dfa([(s_com, e, s_com) for e in [ace, rej, fim]],
                    s_com, f"comunicacao_{id}")

        s_vivo = state("vivo", marked=True)
        chk = self._getE(f"check_vivacidade_{id}")
        A_viv = dfa([(s_vivo, chk, s_vivo)], s_vivo, f"vivacidade_{id}")

        s_bat = state("bateria", marked=True)
        bat = self._getE(f"bateria_baixa_{id}")
        A_bat = dfa([(s_bat, bat, s_bat)], s_bat, f"bateria_{id}")

        self.Uavmodels += [A_com, A_viv, A_bat]
        show_automaton(A_com)

    # -------------------------------------------------------------------------
    def _automato_mapa(self):
        G = self.G
        id = self.id
        state_vert, initial = {}, None
        for v in G.nodes():
            qv = state(f"{v}", marked=("D0" in str(v)))
            state_vert[v] = qv
            if initial is None and "D0" in str(v):
                initial = qv
        if initial is None:
            initial = next(iter(state_vert.values()))

        trs = []
        for u, v, k, _ in G.edges(keys=True, data=True):
            q1, q2 = state_vert[u], state_vert[v]
            pega_ij, pega_ji, _, _ = self.dict_aresta_eventos[(tuple(sorted((u, v))), k)]
            trs += [(q1, pega_ij, q2), (q2, pega_ji, q1)]

        self.mapa = dfa(trs, initial, f"Mapa_{id}")
        show_automaton(self.mapa)

    # -------------------------------------------------------------------------
    def _automato_bateria_movimento(self):
        G = self.G
        id = self.id
        s_normal = state("bateria_normal", marked=True)
        s_baixa = state("bateria_baixa")

        bat_low = self._getE(f"bateria_baixa_{id}")
        trs = [(s_normal, bat_low, s_baixa), (s_baixa, bat_low, s_baixa)]

        for v in G.nodes():
            if G.nodes[v]["tipo"] == "base":
                evt = self._getE(f"fim_carregar_{v}_{id}")
                trs += [(s_baixa, evt, s_normal), (s_normal, evt, s_normal)]

        for u, v, k, _ in G.edges(keys=True, data=True):
            pega_uw, pega_wu, _, _ = self.dict_aresta_eventos[(tuple(sorted((u, v))), k)]
            trs += [(s_normal, pega_uw, s_normal), (s_normal, pega_wu, s_normal)]
            if G.nodes[u]["tipo"] == "base" or G.nodes[v]["tipo"] == "base":
                trs += [(s_baixa, pega_uw, s_baixa), (s_baixa, pega_wu, s_baixa)]

        self.A_bateria_mov = dfa(trs, s_normal, f"movimento_bateria_{id}")
        show_automaton(self.A_bateria_mov)
