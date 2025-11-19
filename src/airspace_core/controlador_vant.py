#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =================================================================================================
# Imports (Python, Grafo, UltraDES)
# =================================================================================================
from typing import Dict, Tuple, List, Any, Iterable, Optional
import os, sys, re, time, math, random, threading
import networkx as nx
from collections import defaultdict, deque

# --- Caminho p/ achar graph/ ao executar via ROS ou direto ---
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)

from airspace_core.uav_agent import VANT
from graph.gerar_grafo import carregar_grafo_txt  
from ultrades.automata import *
from airspace_core.milp_des import otimizador 

# =================================================================================================
# ### Lógica do Nó ROS ###
# =================================================================================================
import rospy
from std_msgs.msg import String

import os
import re

# Regex para extrair coordenadas
_COORD_RE = re.compile(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)')

def carregar_dimensoes_reais():
    """Carrega as dimensões reais do último mundo gerado - VERSÃO ATUALIZADA"""
    diretorios_busca = [
        "./worlds/dimensoes_reais.txt",
        "../worlds/dimensoes_reais.txt", 
        os.path.expanduser("~/catkin_ws/src/airspace_control/worlds/dimensoes_reais.txt"),
        "dimensoes_reais.txt"
    ]
    
    for dim_path in diretorios_busca:
        path_expandido = os.path.expanduser(dim_path)
        if os.path.exists(path_expandido):
            try:
                with open(path_expandido, 'r') as f:
                    lines = f.readlines()
                    # Inicializar com valores padrão
                    dimensoes = {
                        'STAGE_WIDTH': 200.0,
                        'STAGE_HEIGHT': 66.0,
                        'ORIGINAL_WIDTH': 1239.0,
                        'ORIGINAL_HEIGHT': 409.0,
                        'SCALE_FACTOR': 0.323
                    }
                    
                    for line in lines:
                        line = line.strip()
                        if '=' in line:
                            key, value = line.split('=')
                            key = key.strip()
                            value = value.strip()
                            if key in dimensoes:
                                dimensoes[key] = float(value)
                    
                print(f"[INFO] Dimensões carregadas de {path_expandido}:")
                print(f"       Stage: {dimensoes['STAGE_WIDTH']} x {dimensoes['STAGE_HEIGHT']}")
                print(f"       Original: {dimensoes['ORIGINAL_WIDTH']} x {dimensoes['ORIGINAL_HEIGHT']}")
                print(f"       Escala: {dimensoes['SCALE_FACTOR']}")
                
                return dimensoes
            except Exception as e:
                print(f"[ERRO] Falha ao ler {path_expandido}: {e}")
                continue
    
    print("[WARN] Não encontrou dimensoes_reais.txt, usando padrão 200x66")
    return {
        'STAGE_WIDTH': 200.0,
        'STAGE_HEIGHT': 66.0,
        'ORIGINAL_WIDTH': 1239.0,
        'ORIGINAL_HEIGHT': 409.0,
        'SCALE_FACTOR': 0.323
    }

def carregar_posicoes(caminho_arquivo: str):
    """Função CORRIGIDA - usa a mesma lógica de transformação do gerador de mundos"""
    
    # Carregar dimensões do stage
    dimensoes = carregar_dimensoes_reais()
    STAGE_WIDTH = dimensoes['STAGE_WIDTH']
    STAGE_HEIGHT = dimensoes['STAGE_HEIGHT']
    ORIGINAL_WIDTH = dimensoes['ORIGINAL_WIDTH']
    ORIGINAL_HEIGHT = dimensoes['ORIGINAL_HEIGHT']
    SCALE_FACTOR = dimensoes['SCALE_FACTOR']

    if not os.path.exists(caminho_arquivo):
        print(f"[ERRO] Arquivo de grafo não encontrado: {caminho_arquivo}")
        return {}
    
    # Ler arquivo de grafo
    nodes_data = []
    
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        _ = f.readline()  # cabeçalho
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            
            partes = linha.split(",", 3)
            if len(partes) < 3:
                continue
            
            label = partes[1].strip()
            posicao_raw = partes[2].strip()
            
            m = _COORD_RE.match(posicao_raw) or _COORD_RE.search(linha)
            if not m:
                continue
                
            x_do_grafo = float(m.group(1))
            y_do_grafo = float(m.group(2))
            
            nodes_data.append((label, x_do_grafo, y_do_grafo))

    if not nodes_data:
        print("[ERRO] Nenhuma coordenada válida encontrada no arquivo")
        return {}

    print(f"[DEBUG] Transformação de coordenadas:")
    print(f"  - Dimensões Stage: {STAGE_WIDTH} x {STAGE_HEIGHT}")
    print(f"  - Dimensões Original: {ORIGINAL_WIDTH} x {ORIGINAL_HEIGHT}") 
    print(f"  - Fator de Escala: {SCALE_FACTOR}")

    posicoes = {}
    
    for label, x_do_grafo, y_do_grafo in nodes_data:
        # **TRANSFORMAÇÃO CONSISTENTE**: A mesma usada no gerador de mundos
        # 1. Escalar para o Stage usando o mesmo fator
        x_stage = x_do_grafo * SCALE_FACTOR
        y_stage = y_do_grafo * SCALE_FACTOR
        
        # 2. **INVERSÃO DO Y** para o Stage (origem no canto inferior esquerdo)
        y_stage_final = STAGE_HEIGHT - y_stage
        
        # 3. Garantir que está dentro dos limites
        x_stage_final = max(0, min(x_stage, STAGE_WIDTH))
        y_stage_final = max(0, min(y_stage_final, STAGE_HEIGHT))
        
        #print(f"[DEBUG] {label}: ({x_do_grafo}, {y_do_grafo}) -> ({x_stage_final:.1f}, {y_stage_final:.1f})")
        
        posicoes[label] = (label, (x_stage_final, y_stage_final))
    
    print(f"[INFO] Carregadas {len(posicoes)} posições do grafo")
    
    return posicoes

# =================================================================================================
# Classe 1: Modelo genérico (sem sufixo em eventos/estados)
# =================================================================================================

class GenericVANTModel:
    """
    Modelo DES genérico (sem sufixo _{id}) para um grafo e nó inicial.
    Constrói plantas e especificações e permite calcular/salvar supervisores uma única vez.
    """
    
    # ----------------------------- Utilitários internos -----------------------------
    _COORD_RE = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")
    
    @staticmethod
    def _tipo_norm(x: Any) -> str:
        return str(x).strip().upper()
    
    @staticmethod
    def _to_multidigraph_dirigido(G_undirected: nx.Graph) -> nx.MultiDiGraph:
        H = nx.MultiDiGraph()
        H.add_nodes_from(((n, d) for n, d in G_undirected.nodes(data=True)))
        for u, v, d in G_undirected.edges(data=True):
            H.add_edge(u, v, key=0, **(d or {}))
            H.add_edge(v, u, key=0, **(d or {}))
        return H

    # ----------------------------- Construtor -----------------------------
    def __init__(self, grafo_txt: str, init_node: str):
        G_in, _ = carregar_grafo_txt(grafo_txt)
        self.G: nx.MultiDiGraph = self._to_multidigraph_dirigido(G_in)
        self.init_node: str = init_node
        self.grafo_txt: str = grafo_txt
        self.posicoes: Dict[str, Tuple[float, float]] = carregar_posicoes(grafo_txt)
        self.posicao_evento: Dict[str, Tuple[Any, Tuple[float, float]]] = {}
        self.dict_aresta_eventos: Dict[Tuple[Tuple[str, str], Any], Tuple[Any, Any, Any, Any]] = {}
        self.state_vertices: Dict[Any, Any] = {}
        self.eventos: Dict[str, Any] = self._gerar_alfabeto_generico()
        self.plantas: List[Any] = []
        self.specs: List[Any] = []
        self.Dicionario_Automatos: Dict[str, Any] = {}
        self.custos_estado_atomico: Dict[str, Tuple[float, float, float]] = {} # (E, Tf, D)
        self.CUSTO_TEMPO_D = 10
        # Construir todos os autômatos
        self._automato_movimento()
        self._automatos_arestas()
        self._automato_modos()
        self._modelos_suporte()
        self._automato_trabalho()
        self._automato_mapa()
        self._automato_bateria_movimento()
        self._automatos_localizacao_tarefas()
        self._automato_tarefa_completa()
        
        # Inicializar custos APÓS construir todos os autômatos
        self._inicializar_custos_estados()
        self.supervisor_mono=None
        self.supervisor_mono=self.compute_monolithic_supervisor()
        self.dicionario_custos_supervisor=self.criar_dicionario_custo_supervisor()

    # ------------------------- Métodos de Cálculo de Distância e Custos -------------------------

    def _extrair_coordenadas_no(self, no: str):
        """
        Tenta extrair (x, y) de self.posicoes[no] de forma robusta.

        Aceita formatos:
        - posicoes[no] = (x, y)
        - posicoes[no] = (algo, (x, y))
        - se não conseguir interpretar, retorna None.
        """
        if not hasattr(self, "posicoes") or no not in self.posicoes:
            return None

        val = self.posicoes[no]

        # Caso mais simples: (x, y)
        if isinstance(val, tuple) and len(val) == 2:
            a, b = val
            # (x, y) direto
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(a), float(b)
            # (algo, (x, y))
            if isinstance(b, tuple) and len(b) == 2:
                x, y = b
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return float(x), float(y)

        # Se não reconhecer, retorna None sem quebrar o sistema
        return None

    def _calcular_distancia_entre_nos(self, no1: str, no2: str) -> float:
        """
        Calcula a distância euclidiana entre dois nós, baseada nas posições do Stage.

        - Usa um cache interno para não recalcular sempre.
        - Se qualquer nó não tiver posição válida, retorna 1.0 como fallback.
        """
        if not hasattr(self, "_dist_cache"):
            self._dist_cache = {}

        chave = tuple(sorted((no1, no2)))
        if chave in self._dist_cache:
            return self._dist_cache[chave]

        coord1 = self._extrair_coordenadas_no(no1)
        coord2 = self._extrair_coordenadas_no(no2)

        if coord1 is None or coord2 is None:
            dist = 1.0  # fallback
        else:
            x1, y1 = coord1
            x2, y2 = coord2
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        self._dist_cache[chave] = dist
        return dist

    def _obter_tempo_voo_aresta(self, u: str, v: str) -> float:
        """
        Calcula tempo de voo baseado na distância real entre nós.

        Usa self.velocidade_media se existir; caso contrário, assume 2.0 m/s.
        """
        distancia = self._calcular_distancia_entre_nos(u, v)
        velocidade_media = getattr(self, "velocidade_media", 2.0)  # m/s
        if velocidade_media <= 0:
            velocidade_media = 2.0
        return distancia / velocidade_media

    def _obter_consumo_energia_aresta(self, u: str, v: str) -> float:
        """
        Calcula consumo de energia baseado na distância real.

        Retorna um custo POSITIVO (gasto de energia).
        Usa self.consumo_por_metro se existir; caso contrário, assume 0.1.
        """
        distancia = self._calcular_distancia_entre_nos(u, v)
        consumo_por_metro = getattr(self, "consumo_por_metro", 0.1)
        if consumo_por_metro < 0:
            consumo_por_metro = abs(consumo_por_metro)
        return distancia * consumo_por_metro

    def _precomputar_distancia_para_vertices_especiais(self):
        """
        Pré-computa, para cada nó do grafo, a distância até o vértice especial mais próximo.

        Vértices especiais: FORNECEDOR, CLIENTE, ESTACAO, VERTIPORT.
        A distância é euclidiana nas coordenadas do Stage.
        """
        especiais = []
        for n in self.G.nodes():
            tipo_no = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo_no in {"FORNECEDOR", "CLIENTE", "ESTACAO", "VERTIPORT"}:
                especiais.append(n)

        self._dist_min_especial = {}

        # Se não há vértices especiais, tudo fica com distância 0.0
        if not especiais:
            for n in self.G.nodes():
                self._dist_min_especial[n] = 0.0
            return

        for n in self.G.nodes():
            if n in especiais:
                self._dist_min_especial[n] = 0.0
            else:
                # distância até o especial mais próximo
                d_min = min(self._calcular_distancia_entre_nos(n, s) for s in especiais)
                self._dist_min_especial[n] = d_min

    def _inicializar_custos_estados(self):
        """
        Inicializa custos W = [E, Tf, D] com base em uma filosofia de "custo de oportunidade".

        Dimensões:
        - E (Energia):   >0 = custo;  <0 = incentivo (carregar)
        - Tf (Tempo):    >0 = custo de duração física
        - D (Progresso): >0 = penalidade (ociosidade/atraso); <0 = incentivo de missão
        """

        # Garante dicionário de parâmetros de custo
        if not hasattr(self, "cost_params"):
            self.cost_params = {}

        cp = self.cost_params

        # Valores padrão (se não existirem)
        cp.setdefault("CUSTO_TEMPO_D", 0.5)
        cp.setdefault("CUSTO_MOVIMENTO_E", 0.2)
        cp.setdefault("CUSTO_OPERACIONAL_E", 0.1)
        cp.setdefault("INCENTIVO_CARGA_E", -1.0)
        cp.setdefault("INCENTIVO_COLETA_D", -5.0)
        cp.setdefault("INCENTIVO_ENTREGA_D", -10.0)
        cp.setdefault("PENALIDADE_BATERIA_E", 10.0)
        cp.setdefault("PENALIDADE_BATERIA_D", 10.0)

        CUSTO_TEMPO_D        = cp["CUSTO_TEMPO_D"]
        CUSTO_MOVIMENTO_E    = cp["CUSTO_MOVIMENTO_E"]
        CUSTO_OPERACIONAL_E  = cp["CUSTO_OPERACIONAL_E"]
        INCENTIVO_CARGA_E    = cp["INCENTIVO_CARGA_E"]
        INCENTIVO_COLETA_D   = cp["INCENTIVO_COLETA_D"]
        INCENTIVO_ENTREGA_D  = cp["INCENTIVO_ENTREGA_D"]
        PENALIDADE_BATERIA_E = cp["PENALIDADE_BATERIA_E"]
        PENALIDADE_BATERIA_D = cp["PENALIDADE_BATERIA_D"]

        # Precomputar distâncias para vértices especiais (usado na análise global)
        self._precomputar_distancia_para_vertices_especiais()

        # ------------------------------------------------------------------
        # 1. Base: todos os estados atômicos recebem penalidade de tempo D>0
        # ------------------------------------------------------------------
        self.custos_estado_atomico.clear()

        for nome_automato, automato in self.Dicionario_Automatos.items():
            for estado in states(automato):
                est_str = str(estado)
                self.custos_estado_atomico[est_str] = (
                    0.0,            # E (Energia)
                    0.0,            # Tf (Tempo físico)
                    CUSTO_TEMPO_D   # D (penalidade de passo/ociosidade)
                )

        # ------------------------------------------------------------------
        # 2. Movimento: "Movendo" + arestas ocupadas
        # ------------------------------------------------------------------

        # Estado "Movendo" (automato de movimento)
        if "Movendo" in self.custos_estado_atomico:
            # Aqui Tf é MENOR do que ficar parado (ajustamos 'Parado' via análise global)
            self.custos_estado_atomico["Movendo"] = (
                CUSTO_MOVIMENTO_E,  # E: custo base de energia por se mover
                0.1,                # Tf: custo de tempo em movimento
                CUSTO_TEMPO_D       # D: ainda paga custo de passo
            )

        # Estados ocupados de aresta (ocupado_uv / ocupado_vu)
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            if chave not in self.dict_aresta_eventos:
                continue

            tempo_voo = self._obter_tempo_voo_aresta(u, v)
            consumo_energia = self._obter_consumo_energia_aresta(u, v)

            for estado_ocupado in [f"ocupado_{u}{v}", f"ocupado_{v}{u}"]:
                if estado_ocupado in self.custos_estado_atomico:
                    self.custos_estado_atomico[estado_ocupado] = (
                        consumo_energia,  # E: custo de energia da aresta
                        tempo_voo,        # Tf: duração física
                        CUSTO_TEMPO_D     # D: penalidade base de passo
                    )

        # ------------------------------------------------------------------
        # 3. Localização: nós de carga / nós de trabalho
        # ------------------------------------------------------------------
        for nome_no in self.G.nodes():
            estado_mapa = str(nome_no)  # estado do automato "mapa"
            if estado_mapa not in self.custos_estado_atomico:
                continue

            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))

            if tipo_no in {"ESTACAO", "VERTIPORT"}:
                # Incentivo em E (carregar) mas ainda paga penalidade de passo em D
                self.custos_estado_atomico[estado_mapa] = (
                    INCENTIVO_CARGA_E,  # E: incentivo (negativo)
                    0.0,                # Tf
                    CUSTO_TEMPO_D       # D: penalidade base de tempo
                )
            elif tipo_no in {"FORNECEDOR", "CLIENTE"}:
                # Ficar "hovering" no nó de trabalho custa um pouco de energia
                self.custos_estado_atomico[estado_mapa] = (
                    CUSTO_OPERACIONAL_E,  # E: custo leve
                    0.0,                  # Tf
                    CUSTO_TEMPO_D         # D
                )
            # Nós lógicos (sem tipo) permanecem com (0.0, 0.0, CUSTO_TEMPO_D) aqui;
            # o forte custo por ficar parado neles será tratado no nível do supervisor.

        # ------------------------------------------------------------------
        # 4. Estados de trabalho ativo (modos + workflow)
        # ------------------------------------------------------------------

        # Modos locais: trabalhando_FORNECEDOR / trabalhando_CLIENTE
        for nome_no in self.G.nodes():
            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))
            estado_trab = f"trabalhando_{nome_no}"

            if estado_trab in self.custos_estado_atomico:
                if tipo_no == "FORNECEDOR":
                    self.custos_estado_atomico[estado_trab] = (
                        CUSTO_OPERACIONAL_E,   # E: custo de operar
                        0.0,                   # Tf
                        INCENTIVO_COLETA_D     # D: incentivo forte (negativo)
                    )
                elif tipo_no == "CLIENTE":
                    self.custos_estado_atomico[estado_trab] = (
                        CUSTO_OPERACIONAL_E,
                        0.0,
                        INCENTIVO_ENTREGA_D    # D: incentivo ainda maior
                    )

        # Workflow global (pick/place)
        if "pick" in self.custos_estado_atomico:
            self.custos_estado_atomico["pick"] = (
                0.0,
                0.0,
                INCENTIVO_COLETA_D
            )
        if "place" in self.custos_estado_atomico:
            self.custos_estado_atomico["place"] = (
                0.0,
                0.0,
                INCENTIVO_ENTREGA_D
            )

        # ------------------------------------------------------------------
        # 5. Penalidades fortes (bateria baixa, etc.)
        # ------------------------------------------------------------------
        if "bat_baixa" in self.custos_estado_atomico:
            self.custos_estado_atomico["bat_baixa"] = (
                PENALIDADE_BATERIA_E,  # E: penalidade alta
                0.0,
                PENALIDADE_BATERIA_D   # D: penalidade alta
            )

    def obter_custo_estado_supervisor(self, estado_supervisor) -> Tuple[float, float, float]:
        """
        Calcula custo W = [E, Tf, D] para um estado do supervisor, somando:

        1) custos base dos estados atômicos (self.custos_estado_atomico)
        2) ajustes de contexto:
           - penalidade por estar longe de vértices especiais
           - custo maior de Tf quando o agente está Parado
           - custo muito alto por ficar parado em nó lógico
        """
        cp = self.cost_params

        # Parâmetros adicionais para análise no nível do supervisor
        cp.setdefault("PESO_DISTANCIA_D", 0.05)            # ganho em D por metro de distância
        cp.setdefault("FATOR_DISTANCIA_TAREFA", 2.0)       # reforça distância quando há tarefa ativa
        cp.setdefault("EXTRA_TF_PARADO", 0.2)              # Tf extra por estar Parado
        cp.setdefault("EXTRA_TF_PARADO_LOGICO", 0.5)       # Tf extra por estar Parado em nó lógico
        cp.setdefault("PENALIDADE_NO_LOGICO_D", 2.0)       # D extra por passo parado em nó lógico

        PESO_DISTANCIA_D        = cp["PESO_DISTANCIA_D"]
        FATOR_DISTANCIA_TAREFA  = cp["FATOR_DISTANCIA_TAREFA"]
        EXTRA_TF_PARADO         = cp["EXTRA_TF_PARADO"]
        EXTRA_TF_PARADO_LOGICO  = cp["EXTRA_TF_PARADO_LOGICO"]
        PENALIDADE_NO_LOGICO_D  = cp["PENALIDADE_NO_LOGICO_D"]

        E_total, Tf_total, D_total = 0.0, 0.0, 0.0

        # O estado do supervisor vem como string "s1|s2|...|sn"
        componentes = [c.strip() for c in str(estado_supervisor).split('|') if c.strip()]

        # 1) Soma dos custos base de cada estado atômico
        for estado_componente in componentes:
            if estado_componente in self.custos_estado_atomico:
                E, Tf, D = self.custos_estado_atomico[estado_componente]
                E_total += E
                Tf_total += Tf
                D_total += D

        # 2) Análise de contexto: mapa, movimento, modos, trabalho
        map_node = None
        for c in componentes:
            if c in self.G.nodes:  # componente que coincide com um nó do grafo
                map_node = c
                break

        is_moving  = ("Movendo" in componentes)
        is_stopped = ("Parado" in componentes)

        # Presença de automatos mais ligados à tarefa
        has_trabalho_modos = any(c.startswith("trabalhando_") for c in componentes)
        has_tarefa_completa = any(c.startswith("pode_sair_") for c in componentes)
        has_workflow = any(c in {"pick", "place"} for c in componentes)
        # Usamos um flag geral de "contexto de tarefa"
        contexto_tarefa = has_trabalho_modos or has_tarefa_completa or has_workflow

        # 2.1) Penalidade por distância a vértices especiais
        if map_node is not None and hasattr(self, "_dist_min_especial"):
            dist = self._dist_min_especial.get(map_node, 0.0)

            if dist > 0.0:
                ganho_dist = PESO_DISTANCIA_D
                if contexto_tarefa:
                    # Se há tarefa em jogo, ficar longe de vértices especiais é ainda pior
                    ganho_dist *= FATOR_DISTANCIA_TAREFA

                D_total += ganho_dist * dist

                # Se está Parado longe de vértice especial, aumenta também Tf
                if is_stopped:
                    Tf_total += EXTRA_TF_PARADO * (1.0 + dist)

        # 2.2) Penalidade forte por ficar parado em nó lógico (sem tipo)
        if map_node is not None:
            tipo_no = self._tipo_norm(self.G.nodes[map_node].get("tipo", ""))

            if tipo_no == "" and is_stopped:
                # Nó lógico + agente parado: extremamente indesejável
                D_total += PENALIDADE_NO_LOGICO_D
                Tf_total += EXTRA_TF_PARADO_LOGICO

        # 2.3) Mesmo sem nó lógico, estar Parado deve ter Tf maior que Movendo
        #      (se ainda não foi ajustado acima)
        if is_stopped and map_node is None:
            Tf_total += EXTRA_TF_PARADO

        return (E_total, Tf_total, D_total)

    def criar_dicionario_custo_supervisor(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Gera um dicionário mapeando cada estado do supervisor ao seu custo total W=[E, Tf, D].
        Deve ser chamado APÓS compute_monolithic_supervisor().
        """
        if self.supervisor_mono is None:
            raise ValueError(
                "O supervisor monolítico não foi calculado. "
                "Chame 'compute_monolithic_supervisor()' primeiro."
            )

        custos_supervisor: Dict[str, Tuple[float, float, float]] = {}

        for estado_supervisor in states(self.supervisor_mono):
            estado_str = str(estado_supervisor)
            custo_combinado = self.obter_custo_estado_supervisor(estado_str)
            custos_supervisor[estado_str] = custo_combinado

        self.dicionario_custos_supervisor = custos_supervisor
        print(f"[INFO] Dicionário de custos criado para {len(custos_supervisor)} estados do supervisor.")

        return custos_supervisor

    def atualizar_custo_estado_supervisor(
        self,
        estado_supervisor_str: str,
        novo_custo_vetor: Tuple[float, float, float]
    ):
        """
        Atualiza o vetor de custos [E, Tf, D] para um estado específico no dicionário de custos do supervisor.
        """
        # 1. Verificar Inicialização do Dicionário
        if not hasattr(self, 'dicionario_custos_supervisor') or self.dicionario_custos_supervisor is None:
            raise AttributeError(
                "O dicionário de custos do supervisor (self.dicionario_custos_supervisor) "
                "não foi inicializado. Chame 'criar_dicionario_custo_supervisor()' primeiro."
            )

        # 2. Validar o Formato do Custo
        if not isinstance(novo_custo_vetor, tuple) or len(novo_custo_vetor) != 3:
            raise ValueError(
                f"O novo custo deve ser uma tupla de 3 floats (E, Tf, D). Recebido: {novo_custo_vetor}"
            )

        # 3. Atualizar o Custo no Dicionário
        if estado_supervisor_str in self.dicionario_custos_supervisor:

            custo_antigo = self.dicionario_custos_supervisor[estado_supervisor_str]

            # Sobrescreve o custo no dicionário
            self.dicionario_custos_supervisor[estado_supervisor_str] = novo_custo_vetor

            print(f"[INFO] 📝 Custo do estado '{estado_supervisor_str}' atualizado com sucesso.")
            print(f"       Custo Antigo (E, Tf, D): {custo_antigo}")
            print(f"       Novo Custo (E, Tf, D):   {novo_custo_vetor}")

        else:
            raise KeyError(
                f"O estado '{estado_supervisor_str}' não foi encontrado no dicionário de custos do supervisor."
            )

    def atualizar_parametros_custo(
        self,
        consumo_por_metro: float = None,
        velocidade_media: float = None,
        ganho_carregamento: float = None
    ):
        """
        Atualiza parâmetros físicos (velocidade, consumo) e de custo de carga, e recalcula custos.

        - consumo_por_metro: energia gasta por metro (E > 0)
        - velocidade_media:  velocidade de cruzeiro (m/s)
        - ganho_carregamento: módulo do incentivo em E para nós de carga
        """
        if not hasattr(self, "cost_params"):
            self.cost_params = {}

        updated = False

        if consumo_por_metro is not None:
            self.consumo_por_metro = float(abs(consumo_por_metro))
            updated = True

        if velocidade_media is not None:
            self.velocidade_media = float(abs(velocidade_media)) if velocidade_media != 0 else 2.0
            updated = True

        if ganho_carregamento is not None:
            # Interpreta como "quão forte é o incentivo de estar carregando"
            self.cost_params["INCENTIVO_CARGA_E"] = -abs(ganho_carregamento)
            updated = True

        if updated:
            print("[INFO] Parâmetros físicos/custos atualizados - recalculando custos de estados...")
            # Zera caches de distância (se houver)
            if hasattr(self, "_dist_cache"):
                self._dist_cache.clear()
            # Recalcula custos por estado atômico
            self._inicializar_custos_estados()
            # Opcional: se já existe supervisor, recalcule dicionário de custos globais
            if hasattr(self, "supervisor_mono") and self.supervisor_mono is not None:
                self.criar_dicionario_custo_supervisor()

    # ------------------------- Acesso rápido -------------------------
    def ev(self, nome: str) -> Any:
        return self.eventos[nome]
    
    # ------------------------- Geração do Alfabeto (sem _{id}) -------------------------
    def _gerar_alfabeto_generico(self) -> Dict[str, Any]:
        G = self.G
        eventos: Dict[str, Any] = {}
        # 1) Eventos de aresta (pega/libera) — dirigidos
        for u, v, k, data in G.edges(keys=True, data=True):
            for nome in (f"pega_{u}{v}", f"pega_{v}{u}", f"libera_{u}{v}", f"libera_{v}{u}"):
                if nome not in eventos:
                    ctrl = not nome.startswith("libera_")
                    eventos[nome] = event(nome, controllable=ctrl)
            e_uv = f"pega_{u}{v}"; e_vu = f"pega_{v}{u}"
            if v in self.posicoes:
                self.posicao_evento[e_uv] = (eventos[e_uv], self.posicoes[v])
            if u in self.posicoes:
                self.posicao_evento[e_vu] = (eventos[e_vu], self.posicoes[u])
        # 2) Trabalho (FORNECEDOR/CLIENTE)
        for n in G.nodes():
            tipo = self._tipo_norm(G.nodes[n].get("tipo", ""))
            if tipo in {"FORNECEDOR", "CLIENTE"}:
                ini = f"comeca_trabalho_{n}"; fim = f"fim_trabalho_{n}"
                if ini not in eventos: eventos[ini] = event(ini, controllable=True)
                if fim not in eventos: eventos[fim] = event(fim, controllable=False)
        # 3) Carregamento (ESTACAO, VERTIPORT)
        for n in G.nodes():
            tipo = self._tipo_norm(G.nodes[n].get("tipo", ""))
            if tipo in {"ESTACAO"}:
                ini = f"carregar_{n}"; fim = f"fim_carregar_{n}"
                if ini not in eventos: eventos[ini] = event(ini, controllable=True)
                if fim not in eventos: eventos[fim] = event(fim, controllable=False)
        # 4) Globais
        for nome, ctrl in [
            ("aceita_tarefa", True), ("rejeita_tarefa", True),
            ("termina_tarefa", True), ("check_vivacidade", True),
            ("bateria_baixa", False),
        ]:
            if nome not in eventos:
                eventos[nome] = event(nome, controllable=ctrl)
        return eventos
    
    # --------------------------------- Plantas ---------------------------------
    def _automato_movimento(self):
        Parado = state("Parado", marked=True); Movendo = state("Movendo")
        trs = []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            pega_uv = self.ev(f"pega_{u}{v}"); pega_vu = self.ev(f"pega_{v}{u}")
            libera_uv = self.ev(f"libera_{u}{v}"); libera_vu = self.ev(f"libera_{v}{u}")
            if chave not in self.dict_aresta_eventos:
                self.dict_aresta_eventos[chave] = (pega_uv, pega_vu, libera_uv, libera_vu)
            trs.extend([
                (Parado, pega_uv, Movendo), (Movendo, libera_uv, Parado),
                (Parado, pega_vu, Movendo), (Movendo, libera_vu, Parado),
            ])
        A = dfa(trs, Parado, "movimento")
        self.Dicionario_Automatos["movimento"] = A
        self.specs.append(A)
    
    def _automatos_arestas(self):
        vistos = set()
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            if chave in vistos: continue
            vistos.add(chave)
            pega_uv, pega_vu, libera_uv, libera_vu = self.dict_aresta_eventos[chave]
            livre_1 = state(f"livre_{u}{v}", marked=True); ocupado_1 = state(f"ocupado_{u}{v}")
            A1 = dfa([(livre_1, pega_uv, ocupado_1), (ocupado_1, libera_uv, livre_1)], livre_1, f"aresta_{u}{v}_{k}")
            livre_2 = state(f"livre_{v}{u}", marked=True); ocupado_2 = state(f"ocupado_{v}{u}")
            A2 = dfa([(livre_2, pega_vu, ocupado_2), (ocupado_2, libera_vu, livre_2)], livre_2, f"aresta_{v}{u}_{k}")
            self.plantas.extend([A1, A2])
            self.Dicionario_Automatos[f"aresta_{u}{v}_{k}"] = A1
            self.Dicionario_Automatos[f"aresta_{v}{u}_{k}"] = A2
    
    def _automato_modos(self):
        geral = state("geral", marked=True); trs = []
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"FORNECEDOR", "CLIENTE"}:
                s_trab = state(f"trabalhando_{n}")
                e_ini = self.ev(f"comeca_trabalho_{n}"); e_fim = self.ev(f"fim_trabalho_{n}")
                trs.append((geral, e_ini, s_trab)); trs.append((s_trab, e_fim, geral))
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"ESTACAO"}:
                s_c = state(f"carregando_{n}")
                e_ini = self.ev(f"carregar_{n}"); e_fim = self.ev(f"fim_carregar_{n}")
                trs.append((geral, e_ini, s_c)); trs.append((s_c, e_fim, geral))
        A = dfa(trs, geral, "modos")
        self.Dicionario_Automatos["modos"] = A
        self.plantas.append(A)
    
    def _modelos_suporte(self):
        s_com = state("com_ok", marked=True)
        Acom = dfa([
            (s_com, self.ev("aceita_tarefa"), s_com),
            (s_com, self.ev("rejeita_tarefa"), s_com),
            (s_com, self.ev("termina_tarefa"), s_com),
        ], s_com, "comunicacao")
        self.Dicionario_Automatos["comunicacao"] = Acom
        s_vivo = state("vivo", marked=True)
        Aviv = dfa([(s_vivo, self.ev("check_vivacidade"), s_vivo)], s_vivo, "vivacidade")
        self.Dicionario_Automatos["vivacidade"] = Aviv
        s_bat = state("bat", marked=True)
        Abat = dfa([(s_bat, self.ev("bateria_baixa"), s_bat)], s_bat, "bateria")
        self.Dicionario_Automatos["bateria"] = Abat
        self.plantas.append(Abat)
    
    def _automato_mapa(self):
        initial = None
        for n in self.G.nodes():
            s = state(str(n), marked=(n == self.init_node))
            self.state_vertices[n] = s
            if n == self.init_node: initial = s
        if initial is None:
            first = next(iter(self.G.nodes())); initial = self.state_vertices[first]
        trs = []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            su = self.state_vertices[u]; sv = self.state_vertices[v]
            trs.append((su, self.ev(f"pega_{u}{v}"), sv))
            trs.append((sv, self.ev(f"pega_{v}{u}"), su))
        A = dfa(trs, initial, "Mapa")
        self.Dicionario_Automatos["mapa"] = A
        self.specs.append(A)
    
    def _automato_bateria_movimento(self):
        s_norm = state("bat_normal", marked=True); s_low  = state("bat_baixa")
        e_low  = self.ev("bateria_baixa")
        trs = [(s_norm, e_low, s_low), (s_low,  e_low, s_low)]
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"ESTACAO"}:
                e_ini = self.ev(f"carregar_{n}")
                trs.extend([(s_low,  e_ini, s_norm), (s_norm, e_ini, s_norm)])
        A = dfa(trs, s_norm, "MovimentoBateria")
        self.Dicionario_Automatos["bat_mov"] = accessible(A)
        self.specs.append(self.Dicionario_Automatos["bat_mov"])
    
    def _automatos_localizacao_tarefas(self):
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo not in {"FORNECEDOR", "CLIENTE", "ESTACAO"}: continue
            s_in  = state(f"dentro_{n}"); s_out = state(f"fora_{n}", marked=True)
            trs = []
            for x in self.G.neighbors(n): # Entrada: libera_{x n}
                trs.append((s_out, self.ev(f"libera_{x}{n}"), s_in))
            for x in self.G.neighbors(n): # Saída: pega_{n x}
                trs.append((s_in, self.ev(f"pega_{n}{x}"), s_out))
            if tipo in {"FORNECEDOR", "CLIENTE"}: # Tarefas locais
                trs.append((s_in, self.ev(f"comeca_trabalho_{n}"), s_in))
            if tipo == "ESTACAO":
                trs.append((s_in, self.ev(f"carregar_{n}"), s_in))
            A = dfa(trs, s_out, f"loc_{n}")
            self.Dicionario_Automatos[f"loc_{n}"] = A
            self.specs.append(A)
   
    def _automato_trabalho(self):
        s_pick   = state("pick", marked=False)
        s_place  = state("place", marked=False)
        s_base   = state("vantport", marked=True)
        
        trs = []
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo in {"FORNECEDOR"}: 
                trs.append((s_base, self.ev(f"comeca_trabalho_{n}"), s_pick))
            if tipo in {"CLIENTE"}: 
                trs.append((s_pick, self.ev(f"comeca_trabalho_{n}"), s_place))
            if tipo in {"VERTIPORT"}:
                for u, v, k, data in self.G.in_edges(n, keys=True, data=True):
                    trs.append((s_place, self.ev(f"pega_{u}{n}"), s_base))

        
        A = dfa(trs, s_base, f"work_flow_{n}")
        self.Dicionario_Automatos[f"work_flow_{n}"] = A
        self.specs.append(A)

    def _automato_tarefa_completa(self):
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            
            # Só consideramos nós que têm tarefas locais
            if tipo not in {"FORNECEDOR", "CLIENTE", "ESTACAO", "VERTIPORT"}: 
                continue

            s_pode = state(f"pode_sair_{n}", marked=True)
            s_trab = state(f"trabalhando_{n}")
            trs = []

            # 1. Eventos de Saída (ES)
            # O agente pode tentar sair se estiver no estado 'pode_sair'
            eventos_saida = [self.ev(f"pega_{n}{x}") for x in self.G.neighbors(n)]
            for e_saida in eventos_saida:
                trs.append((s_pode, e_saida, s_pode)) # Autolaço: permite sair

            # 2. Eventos de Início e Fim de Tarefa (ET_ini e ET_fim)
            e_ini = None; e_fim = None
            if tipo in {"FORNECEDOR", "CLIENTE"}:
                e_ini = self.ev(f"comeca_trabalho_{n}")
                e_fim = self.ev(f"fim_trabalho_{n}")
            elif tipo in {"ESTACAO"}:
                e_ini = self.ev(f"carregar_{n}")
                e_fim = self.ev(f"fim_carregar_{n}")
            
            if e_ini and e_fim:
                # Transição: Inicia o trabalho/carregamento (Vai para o estado restritivo)
                trs.append((s_pode, e_ini, s_trab)) 
                
                # Transição: Permite o autolaço do evento de início no estado de trabalho (opcional)
                trs.append((s_trab, e_ini, s_trab)) 
                
                # Transição: Fim do trabalho/carregamento (Volta ao estado livre)
                trs.append((s_trab, e_fim, s_pode))
                
                # A RESTRIÇÃO PRINCIPAL é a ausência de transição (s_trab, e_saida, ...)
                # O DFA irá automaticamente restringir os eventos de saída (pega_nx) no estado s_trab.
                
            A = dfa(trs, s_pode, f"tarefa_completa_{n}")
            self.Dicionario_Automatos[f"tarefa_completa_{n}"] = A
            self.specs.append(A)

    # ------------------------------- Supervisor / IO -------------------------------
    def compute_monolithic_supervisor(self, force: bool = False) -> Any:
        if self.supervisor_mono is None or force:
            self.supervisor_mono = monolithic_supervisor(self.plantas, self.specs)
        return self.supervisor_mono

# =================================================================================================
# Classe 2: Supervisor + Controle Inteligente por MILP 
# =================================================================================================
class VANTInstance:
    """
    Especializa o supervisor genérico para um VANT específico (id_num),
    incluindo um sistema de controle inteligente baseado em otimização MILP (Janela Deslizante).

    Esta versão garante que a otimização usa eventos com ID (vant_1) e publica no barramento 
    ROS com eventos GENÉRICOS (vant).
    """
    _RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")

    def __init__(self,
                 model: 'GenericVANTModel',
                 id_num: int,
                 supervisor_mono=None,
                 obj_vant=None,
                 enable_ros: bool = False,
                 node_name: str = None):
        
        # ----------------------- Inicialização Base (Supervisor DES) -----------------------
        self.model = model
        self.id = int(id_num)
        self.obj_vant = obj_vant
        self.enable_ros = bool(enable_ros)
        self.name = node_name or f"supervisor_vant_{self.id}"
        self.posicoes = model.posicao_evento 

        # Tarefa ativa / MILP (apenas um por vez)
        self._tarefa_ativa = None           # (fornecedor, cliente)
        self._milp_thread = None
        self._milp_thread_lock = threading.Lock()
        self._planning_horizon = 6         # Horizonte padrão para o MILP (ajuste se quiser)
        self._claimed_tasks = set()

        # 1) Recupera (ou calcula) supervisor genérico
        if supervisor_mono is None:
            if getattr(model, "supervisor_mono", None) is None:
                supervisor_mono = model.compute_monolithic_supervisor() 
            else:
                supervisor_mono = model.supervisor_mono
        self._sup_gen = supervisor_mono

        # 2) Renomeia TODAS as transições do supervisor para evento_{id}
        trs_gen = list(transitions(self._sup_gen))
        renamed_trs = []
        self.event_map = {}      # e_gen (str) -> e_id (str)
        self.rev_event_map = {}  # e_id (str) -> e_gen (str)
        self._event_objects = {} # e_id (str) -> AbstractEvent object

        for (q, e, q2) in trs_gen:
            es = str(e)
            es_id = f"{es}_{self.id}"

            # Reutiliza sempre o MESMO objeto Event para cada nome es_id
            if es_id not in self._event_objects:
                new_event = event(es_id, controllable=is_controllable(e))
                self._event_objects[es_id] = new_event
                self.event_map[es] = es_id
                self.rev_event_map[es_id] = es

            ev_obj = self._event_objects[es_id]
            renamed_trs.append((q, ev_obj, q2))

        # 3) Constrói o DFA id-específico com objetos Event corretos
        self.supervisor = dfa(renamed_trs, initial_state(self._sup_gen), f"sup_id_{self.id}")
        self._trs_id = list(transitions(self.supervisor))
        self._state = initial_state(self.supervisor)

        # 4) (Opcional) ROS — desligado por padrão
        if self.enable_ros:
            import rospy
            from std_msgs.msg import String

            self.ros = rospy.init_node(self.name, anonymous=False)

            # Publicadores de estado do supervisor
            self.pub_state  = rospy.Publisher(f"/{self.name}/state", String, queue_size=10, latch=True)
            self.pub_events = rospy.Publisher(f"/{self.name}/possible_events", String, queue_size=10, latch=True)
            self.pub_marked = rospy.Publisher(f"/{self.name}/is_marked", String, queue_size=10, latch=True)
            self.pub_enabled_events = rospy.Publisher(f"/{self.name}/enabled_events", String, queue_size=10, latch=True)

            # Publisher para eventos de controle (incluindo saída do MILP)
            self.pub_cmd_event = rospy.Publisher("/event", String, queue_size=50)

            # Subscriber padrão para /event (callback simples)
            self.sub_event = rospy.Subscriber("/event", String, self._on_event, queue_size=50)

            # Subscriber de tarefas (NOVO)
            self.sub_tarefas = rospy.Subscriber(
                "/tarefas",
                String,
                self._callback_tarefas,
                queue_size=10
            )

            self.pub_tarefas_claim = rospy.Publisher("/tarefas_claims", String, queue_size=10)
            self.sub_tarefas_claim = rospy.Subscriber(
                "/tarefas_claims",
                String,
                self._callback_tarefas_claim,
                queue_size=10
            )

            rospy.sleep(0.3)
            self._publish_ros()


    # método existente na classe
    def _on_event(self, msg):
        """Recebe eventos do barramento /event (String)."""
        ev = str(msg.data or "")
        if ev == "ping":
            # re-publica o snapshot para o painel
            self._publish_ros()
            return
        # aplica se for do meu id e estiver habilitado no estado
        _ = self.step(ev)  # step já filtra por id e verifica a transição

    # ----------------------- CALLBACK DE TAREFAS (NOVO) -----------------------

    def _callback_tarefas_claim(self, msg):
        """
        Recebe claims de tarefas no formato exatamente igual ao /tarefas:
            'FORNECEDOR_X,CLIENTE_Y'
        e registra que essa tarefa já foi pega por algum VANT.
        """
        raw = str(msg.data or "").strip()
        if not raw:
            return
        self._claimed_tasks.add(raw)


    def _callback_tarefas(self, msg):
        """
        Recebe tarefas no formato:
            'FORNECEDOR_X,CLIENTE_Y'

        Protocolo:
          - Se já existe claim para essa tarefa em _claimed_tasks, ignora.
          - Caso contrário, espera um pequeno delay aleatório e verifica de novo.
          - Se ainda não houver claim, este VANT faz o claim, marca como tarefa ativa
            e dispara o MILP.
        """
        if not self.enable_ros:
            return

        import rospy
        from std_msgs.msg import String
        import random

        raw = str(msg.data or "").strip()
        if not raw:
            return

        # Ex.: "FORNECEDOR_0,CLIENTE_0"
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            rospy.logwarn(f"[{self.name}] Formato inválido de tarefa recebida: '{raw}'. Esperado 'FORNECEDOR_X,CLIENTE_Y'.")
            return

        fornecedor, cliente = parts[0], parts[1]

        # Se a tarefa já foi claimada por qualquer VANT, ignora
        if raw in self._claimed_tasks:
            rospy.loginfo(f"[{self.name}] Tarefa '{raw}' já foi claimada. Ignorando.")
            return

        # Se este VANT já está ocupado com outra tarefa, ignora
        if self._tarefa_ativa is not None:
            rospy.loginfo(f"[{self.name}] Já possuo tarefa ativa {self._tarefa_ativa}. Ignorando '{raw}'.")
            return

        # Pequeno atraso aleatório para evitar empates (em ms / décimos de segundo)
        delay = random.uniform(0.0, 0.5)
        rospy.sleep(delay)

        # Depois do delay, checa de novo se alguém já claimou
        if raw in self._claimed_tasks:
            rospy.loginfo(f"[{self.name}] Após delay, tarefa '{raw}' já foi claimada por outro VANT. Ignorando.")
            return

        # Agora este VANT faz o claim
        self._tarefa_ativa = (fornecedor, cliente)
        self._claimed_tasks.add(raw)
        rospy.loginfo(f"[{self.name}] Tarefa recebida e CLAIMADA: {self._tarefa_ativa}.")

        # Notifica os outros VANTs do claim
        self.pub_tarefas_claim.publish(String(data=raw))

        # Inicia thread do MILP se não houver uma rodando
        with self._milp_thread_lock:
            if self._milp_thread is None or not self._milp_thread.is_alive():
                self._milp_thread = threading.Thread(
                    target=self._run_milp_for_current_task,
                    daemon=True
                )
                self._milp_thread.start()


    # ----------------------- API LÓGICA PARA TESTE -----------------------
    def state(self):
        return self._state

    def enabled_events(self):
        """Eventos habilitados (já com sufixo _{id}) como strings."""
        s = str(self._state)
        out = []
        for (q, e, _d) in self._trs_id:
            if str(q) == s:
                out.append(str(e))  # Retorna como string para interface externa
        return sorted(set(out))

    def _should_process(self, ev: str) -> bool:
        """
        Regras:
          - Só processa eventos terminando com _{id} do próprio agente.
          - Eventos 'puros' (sem sufixo) e eventos de outro id são ignorados.
        """
        m = self._RE_SUFFIX.match(ev)
        if not m:
            return False
        return (int(m.group(2)) == self.id)


    def step(self, ev: str) -> bool:
        """
        Tenta aplicar o evento 'ev'. Retorna True se transicionou, False caso contrário.
        Recebe string como entrada, converte para objeto Event internamente.

        NOVO: se a transição aplicada for um 'libera_*_id' e existir uma
        tarefa ativa, dispara um novo MILP (receding horizon).
        """
        if not self._should_process(ev):
            return False
        
        # Converter string para objeto Event correspondente
        event_obj = self._event_objects.get(ev)
        if event_obj is None:
            return False

        s = str(self._state)
        transicionou = False

        for (q, e, d) in self._trs_id:
            if str(q) == s and e == event_obj:
                self._state = d
                transicionou = True
                if self.enable_ros:
                    self._publish_ros()
                break

        if not transicionou:
            return False

        # ------------------ GATILHO DE REPLANEJAMENTO (MPC) ------------------
        # Se ainda há tarefa ativa e o evento é uma liberação deste VANT,
        # disparamos um novo MILP para calcular o próximo trecho da rota.
        if self._tarefa_ativa is not None and ev.startswith("libera_"):
            import threading
            import rospy

            rospy.loginfo(f"[{self.name}] Evento de liberação '{ev}' aplicado. Replanejando (novo MILP) para tarefa ativa {self._tarefa_ativa}.")

            with self._milp_thread_lock:
                if self._milp_thread is None or not self._milp_thread.is_alive():
                    self._milp_thread = threading.Thread(
                        target=self._run_milp_for_current_task,
                        daemon=True
                    )
                    self._milp_thread.start()

        # ---------------------------------------------------------------------
        return True
    def _run_milp_for_current_task(self):
        """
        Executa o otimizador MILP para a tarefa ativa atual e,
        se obtiver uma sequência de eventos, aplica o PRIMEIRO
        evento controlável que esteja habilitado e publica em /event.

        Importante:
        - Em caso de sucesso, a tarefa permanece ativa (_tarefa_ativa NÃO é limpa).
        - Em caso de falha (sequência vazia ou nenhum evento habilitado), a tarefa é abortada.
        """
        if not self.enable_ros:
            return

        import rospy
        from std_msgs.msg import String

        # Se não há tarefa ativa, não há o que otimizar
        if self._tarefa_ativa is None:
            return

        tarefa = self._tarefa_ativa
        fornecedor, cliente = tarefa

        try:
            # Horizonte
            H = self._planning_horizon

            # Eventos de interesse (GENÉRICOS)
            eventos_interesse_gen = [
                f"comeca_trabalho_{fornecedor}",
                f"comeca_trabalho_{cliente}",
            ]
            # Versão com ID (o autômato que passamos é o id-específico)
            eventos_interesse_id = [f"{nm}_{self.id}" for nm in eventos_interesse_gen]

            # Nenhum evento proibido adicional, por enquanto
            eventos_proibidos_id = []

            # Estado atual do supervisor (id-específico)
            estado_inicial = self._state

            # Dicionário de custos vem do modelo genérico
            cost_dict = getattr(self.model, "dicionario_custos_supervisor", {})

            rospy.loginfo(f"[{self.name}] Iniciando MILP para tarefa {tarefa} com H={H}.")

            # Chamada ao otimizador (já com lock global dentro do módulo milp_des)
            event_seq, status = otimizador(
                self.supervisor,
                estado_inicial,
                H,
                cost_dict,
                eventos_interesse_id,
                eventos_proibidos_id
            )

            rospy.loginfo(f"[{self.name}] MILP retornou status={status}, seq={event_seq}.")

            # Se não veio sequência, aborta tarefa
            if not event_seq:
                rospy.logwarn(
                    f"[{self.name}] Sequência de eventos vazia. "
                    f"Abortando tarefa {tarefa}."
                )
                

            # Eventos habilitados no estado atual (id-específicos)
            enabled = set(self.enabled_events())

            selected = None
            for ev_name in event_seq:
                if ev_name in enabled:
                    ev_obj = self._event_objects.get(ev_name)
                    # Garante que é evento controlável
                    if ev_obj is not None and is_controllable(ev_obj):
                        selected = ev_name
                        break

            if selected is None:
                rospy.logwarn(
                    f"[{self.name}] Nenhum evento controlável da sequência MILP está habilitado. "
                    f"Abortando tarefa {tarefa}."
                )
                self._tarefa_ativa = None
                return

            # Publica o evento selecionado em /event
            rospy.loginfo(f"[{self.name}] Publicando evento MILP selecionado: {selected}")
            self.pub_cmd_event.publish(String(data=selected))

            # IMPORTANTE: não limpamos _tarefa_ativa aqui.
            # A tarefa continua ativa. Um novo MILP será disparado
            # quando o step() processar um evento de liberação (libera_*) deste VANT.

        except Exception as e:
            rospy.logerr(f"[{self.name}] Erro executando MILP para tarefa {tarefa}: {e}")
            # Em caso de erro inesperado, aborta a tarefa para não ficar preso
            self._tarefa_ativa = None

    # --------------------------- ROS (opcional) ---------------------------
    def _publish_ros(self):
        """Publica estado/eventos/hint de marcado quando enable_ros=True."""
        import rospy
        from std_msgs.msg import String
        self.pub_state.publish(str(self._state))
        evs = ",".join(self.enabled_events())
        self.pub_events.publish(evs)
        self.pub_enabled_events.publish(evs)
        self.pub_marked.publish("True" if is_marked(self._state) else "False")

    # Extra útil em integrações físicas
    def to_generic(self, ev_with_id: str) -> str:
        return self.rev_event_map.get(ev_with_id, ev_with_id)

    def run(self):
        """Loop principal (apenas quando ROS está habilitado)."""
        if not self.enable_ros:
            return
        import rospy
        rospy.spin()














