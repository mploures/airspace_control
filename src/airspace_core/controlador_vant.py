#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Any
import re
import networkx as nx
import os
import sys

# adiciona a raiz do pacote para achar graph/
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)

from graph.gerar_grafo import carregar_grafo_txt  # retorna (G, pos)

# UltraDES API
from ultrades.automata import *  # event, state, dfa, accessible, etc.

# ----------------------------- Util: carregar posições -----------------------------
_COORD_RE = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")

def carregar_posicoes(caminho_arquivo: str) -> Dict[str, Tuple[float, float]]:
    """
    Lê o arquivo do grafo (formato CSV "solto") e retorna {label: (x, y)} como float.
    Robusto a colunas variáveis após 'posicao'.
    """
    posicoes: Dict[str, Tuple[float, float]] = {}
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        _ = f.readline()  # pula cabeçalho
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            partes = linha.split(",", 3)
            if len(partes) < 3:
                continue  # linha inválida
            label = partes[1].strip()
            posicao_raw = partes[2].strip()

            m = _COORD_RE.match(posicao_raw) or _COORD_RE.search(linha)
            if not m:
                raise ValueError(f"Não foi possível extrair coordenadas de: {linha}")

            x = float(m.group(1))
            y = float(m.group(2))
            posicoes[label] = (x, y)

    return posicoes


def _to_multidigraph_dirigido(G_undirected: nx.Graph) -> nx.MultiDiGraph:
    """
    Converte Graph não-dirigido (com atributo 'tipo' nos nós) em MultiDiGraph dirigido,
    duplicando cada aresta u-v como u->v e v->u (key=0).
    """
    H = nx.MultiDiGraph()
    H.add_nodes_from(((n, d) for n, d in G_undirected.nodes(data=True)))
    for u, v, d in G_undirected.edges(data=True):
        H.add_edge(u, v, key=0, **(d or {}))
        H.add_edge(v, u, key=0, **(d or {}))
    return H


# ================================== Classe Controlador ==================================
class ControladorVANT:
    # ----------------------------- Construtor -----------------------------
    def __init__(self, id_vant: int, obj_vant: Any, grafo_txt: str, init_node: str):
        # Identidade
        self.id_num: int = int(id_vant)
        self.name: str = f"vant_{self.id_num}"
        self._suf: str = f"_{self.id_num}"

        # Físico e grafo
        self.vant = obj_vant
        G_in, _pos = carregar_grafo_txt(grafo_txt)   # (G, pos)
        self.G: nx.MultiDiGraph = _to_multidigraph_dirigido(G_in)
        self.init_node: str = init_node

        # Armazenamento
        self.posicoes: Dict[str, Tuple[float, float]] = carregar_posicoes(grafo_txt)
        self.posicao_evento: Dict[str, Tuple[Any, Tuple[float, float]]] = {}

        self.dict_aresta_eventos: Dict[Tuple[Tuple[str, str], Any], Tuple[Any, Any, Any, Any]] = {}
        self.state_vertices: Dict[Any, Any] = {}
        self.Dicionario_Automatos: Dict[str, Any] = {}
        self.plantas=[]
        self.specs=[]

        # Eventos (alfabeto) — depois que posicoes/posicao_evento existem
        self.eventos: Dict[str, Any] = self.gerarAlfabeto()

        # Modelos
        self._automato_movimento()
        self._automatos_arestas()
        self._automato_modos()
        self._modelos_suporte()
        self._automato_mapa()                   # especificação
        self._automato_bateria_movimento()      # especificação

    # ------------------------- Aux -------------------------
    def _tipo_norm(self, t: Any) -> str:
        # manter apenas 5 tipos oficiais em UPPER
        return str(t).strip().upper()

    def _ev(self, nome: str) -> Any:
        return self.eventos[nome]

    # ------------------------- Geração do Alfabeto -------------------------
    def gerarAlfabeto(self) -> Dict[str, Any]:
        """
        Retorna um dicionário {str(evento): evento} com todos os eventos usados
        no sistema, nomeados com sufixo do VANT (ex: pega_D0L0_0).
        SOMENTE via sufixo "_{id}" (id é número).
        """
        G = self.G
        idn = self.id_num
        eventos: Dict[str, Any] = {}

        # === 1) Eventos de arestas (pega/libera) ===
        for u, v, k, data in G.edges(keys=True, data=True):
            for nome in (
                f"pega_{u}{v}_{idn}",
                f"pega_{v}{u}_{idn}",
                f"libera_{u}{v}_{idn}",
                f"libera_{v}{u}_{idn}",
            ):
                if nome not in eventos:
                    ctrl = not nome.startswith("libera_")  # libera_* é não controlável
                    eventos[nome] = event(nome, controllable=ctrl)

            # posiciona os eventos de "pega" no nó de destino
            e_uv = f"pega_{u}{v}_{idn}"
            e_vu = f"pega_{v}{u}_{idn}"
            if v in self.posicoes:
                self.posicao_evento[e_uv] = (eventos[e_uv], self.posicoes[v])
            if u in self.posicoes:
                self.posicao_evento[e_vu] = (eventos[e_vu], self.posicoes[u])

        # === 2) Trabalho (FORNECEDOR/CLIENTE) ===
        for n in G.nodes():
            tipo = self._tipo_norm(G.nodes[n].get("tipo", ""))
            if tipo in {"FORNECEDOR", "CLIENTE"}:
                ini = f"comeca_trabalho_{n}_{idn}"
                fim = f"fim_trabalho_{n}_{idn}"
                if ini not in eventos:
                    eventos[ini] = event(ini, controllable=True)
                if fim not in eventos:
                    eventos[fim] = event(fim, controllable=False)

        # === 3) Carregamento (apenas ESTACAO e VERTIPORT) ===
        for n in G.nodes():
            tipo = self._tipo_norm(G.nodes[n].get("tipo", ""))
            if tipo in {"ESTACAO", "VERTIPORT"}:
                ini = f"carregar_{n}_{idn}"
                fim = f"fim_carregar_{n}_{idn}"
                if ini not in eventos:
                    eventos[ini] = event(ini, controllable=True)
                if fim not in eventos:
                    eventos[fim] = event(fim, controllable=False)

        # === 4) Globais ===
        for nome, ctrl in [
            (f"aceita_tarefa_{idn}", True),
            (f"rejeita_tarefa_{idn}", True),
            (f"termina_tarefa_{idn}", True),
            (f"check_vivacidade_{idn}", True),
            (f"bateria_baixa_{idn}", False),
        ]:
            if nome not in eventos:
                eventos[nome] = event(nome, controllable=ctrl)

        return eventos

    # ----------------------- Plantas: Movimento --------------------------
    def _automato_movimento(self):
        Parado = state(f"Parado{self._suf}", marked=True)
        Movendo = state(f"Movendo{self._suf}")

        trs = []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            pega_uv = self._ev(f"pega_{u}{v}_{self.id_num}")
            pega_vu = self._ev(f"pega_{v}{u}_{self.id_num}")
            libera_uv = self._ev(f"libera_{u}{v}_{self.id_num}")
            libera_vu = self._ev(f"libera_{v}{u}_{self.id_num}")

            if chave not in self.dict_aresta_eventos:
                self.dict_aresta_eventos[chave] = (pega_uv, pega_vu, libera_uv, libera_vu)

            trs.extend([
                (Parado, pega_uv, Movendo),
                (Movendo, libera_uv, Parado),
                (Parado, pega_vu, Movendo),
                (Movendo, libera_vu, Parado),
            ])

        A = dfa(trs, Parado, f"movimento{self._suf}")
        self.Dicionario_Automatos["movimento"] = A
        self.plantas.append(A)

    # --------------------- Plantas: Arestas (cap=1) ----------------------
    def _automatos_arestas(self):
        vistos = set()
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            if chave in vistos:
                continue
            vistos.add(chave)

            pega_uv, pega_vu, libera_uv, libera_vu = self.dict_aresta_eventos[chave]

            # u -> v
            livre_1 = state(f"livre_{u}{v}{self._suf}", marked=True)
            ocupado_1 = state(f"ocupado_{u}{v}{self._suf}")
            A1 = dfa(
                [(livre_1, pega_uv, ocupado_1),
                 (ocupado_1, libera_uv, livre_1)],
                livre_1, f"aresta_{u}{v}_{k}{self._suf}"
            )

            # v -> u
            livre_2 = state(f"livre_{v}{u}{self._suf}", marked=True)
            ocupado_2 = state(f"ocupado_{v}{u}{self._suf}")
            A2 = dfa(
                [(livre_2, pega_vu, ocupado_2),
                 (ocupado_2, libera_vu, livre_2)],
                livre_2, f"aresta_{v}{u}_{k}{self._suf}"
            )

            self.plantas.extend([A1, A2])
            self.Dicionario_Automatos[f"aresta_{u}{v}_{k}"] = A1
            self.Dicionario_Automatos[f"aresta_{v}{u}_{k}"] = A2

    # ------------------ Planta: Modos (trabalho/carga) -------------------
    def _automato_modos(self):
        geral = state(f"geral{self._suf}", marked=True)
        trs = []

        # movimentação não muda modo
        for u, v, k, data in self.G.edges(keys=True, data=True):
            trs.append((geral, self._ev(f"pega_{u}{v}_{self.id_num}"), geral))
            trs.append((geral, self._ev(f"pega_{v}{u}_{self.id_num}"), geral))

        # Trabalho
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"FORNECEDOR", "CLIENTE"}:
                s_trab = state(f"trabalhando_{n}{self._suf}")
                e_ini = self._ev(f"comeca_trabalho_{n}_{self.id_num}")
                e_fim = self._ev(f"fim_trabalho_{n}_{self.id_num}")
                trs.append((geral, e_ini, s_trab))
                trs.append((s_trab, e_fim, geral))

        # Carregamento
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"ESTACAO", "VERTIPORT"}:
                s_c = state(f"carregando_{n}{self._suf}")
                e_ini = self._ev(f"carregar_{n}_{self.id_num}")
                e_fim = self._ev(f"fim_carregar_{n}_{self.id_num}")
                trs.append((geral, e_ini, s_c))
                trs.append((s_c, e_fim, geral))

        A = dfa(trs, geral, f"modos{self._suf}")
        self.Dicionario_Automatos["modos"] = A
        self.plantas.append(A)

    # ------------------ Plantas: Suporte (com/vivo/bat) ------------------
    def _modelos_suporte(self):
        # Comunicação
        s_com = state(f"com_ok{self._suf}", marked=True)
        Acom = dfa([
            (s_com, self._ev(f"aceita_tarefa_{self.id_num}"), s_com),
            (s_com, self._ev(f"rejeita_tarefa_{self.id_num}"), s_com),
            (s_com, self._ev(f"termina_tarefa_{self.id_num}"), s_com),
        ], s_com, f"comunicacao{self._suf}")
        self.Dicionario_Automatos["comunicacao"] = Acom
        self.plantas.append(Acom)

        # Vivacidade
        s_vivo = state(f"vivo{self._suf}", marked=True)
        Aviv = dfa([(s_vivo, self._ev(f"check_vivacidade_{self.id_num}"), s_vivo)],
                   s_vivo, f"vivacidade{self._suf}")
        self.Dicionario_Automatos["vivacidade"] = Aviv
        self.plantas.append(Aviv)

        # Bateria (monitor)
        s_bat = state(f"bat{self._suf}", marked=True)
        Abat = dfa([(s_bat, self._ev(f"bateria_baixa_{self.id_num}"), s_bat)],
                   s_bat, f"bateria{self._suf}")
        self.Dicionario_Automatos["bateria"] = Abat
        self.plantas.append(Abat)

    # ---------------------- Especificação: Mapa --------------------------
    def _automato_mapa(self):
        # Estado por nó; inicial é o init_node se existir, senão o primeiro
        initial = None
        for n in self.G.nodes():
            s = state(str(n), marked=(n == self.init_node))
            self.state_vertices[n] = s
            if n == self.init_node:
                initial = s
        if initial is None:
            first = next(iter(self.G.nodes()))
            initial = self.state_vertices[first]  # não força marcação aqui

        trs = []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            su = self.state_vertices[u]
            sv = self.state_vertices[v]
            trs.append((su, self._ev(f"pega_{u}{v}_{self.id_num}"), sv))
            trs.append((sv, self._ev(f"pega_{v}{u}_{self.id_num}"), su))

        A = dfa(trs, initial, f"Mapa{self._suf}")
        self.Dicionario_Automatos["mapa"] = A
        self.specs.append(A)

    # ------------- Especificação: Bateria x Movimento --------------------
    def _automato_bateria_movimento(self):
        s_norm = state(f"bat_normal{self._suf}", marked=True)
        s_low  = state(f"bat_baixa{self._suf}")
        e_low  = self._ev(f"bateria_baixa_{self.id_num}")

        trs = [(s_norm, e_low, s_low), (s_low, e_low, s_low)]

        # volta ao normal ao terminar carga em QUALQUER ESTACAO/VERTIPORT
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"ESTACAO", "VERTIPORT"}:
                e_ok = self._ev(f"fim_carregar_{n}_{self.id_num}")
                trs.append((s_low, e_ok, s_norm))
                trs.append((s_norm, e_ok, s_norm))

        # arestas autorizadas em baixa bateria: que tocam ESTACAO/VERTIPORT
        arestas_permitidas_baixa = set()
        for u, v, k, data in self.G.edges(keys=True, data=True):
            tu = self._tipo_norm(self.G.nodes[u].get("tipo", ""))
            tv = self._tipo_norm(self.G.nodes[v].get("tipo", ""))
            if (tu in {"ESTACAO", "VERTIPORT","LOGICO"}) or (tv in {"ESTACAO", "VERTIPORT","LOGICO"}):
                arestas_permitidas_baixa.add((u, v, k))

        for u, v, k, data in self.G.edges(keys=True, data=True):
            e_uv = self._ev(f"pega_{u}{v}_{self.id_num}")
            e_vu = self._ev(f"pega_{v}{u}_{self.id_num}")

            # normal: sempre aceita
            trs.append((s_norm, e_uv, s_norm))
            trs.append((s_norm, e_vu, s_norm))

            # baixa: só se a aresta tocar ESTACAO/VERTIPORT
            if (u, v, k) in arestas_permitidas_baixa:
                trs.append((s_low, e_uv, s_low))
                trs.append((s_low, e_vu, s_low))

        A = dfa(trs, s_norm, f"MovimentoBateria{self._suf}")
        self.Dicionario_Automatos["bat_mov"] = accessible(A)
        self.specs.append(self.Dicionario_Automatos["bat_mov"])
