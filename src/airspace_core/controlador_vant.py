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
        self.CUSTO_TEMPO_D = 0.1
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
        self._automato_fim_de_carga()
        
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
        Inicializa custos dos estados atômicos com valores balanceados que atendam aos requisitos:
        1. Realizar tarefa tem maior incentivo
        2. Ficar parado é penalizado
        3. Visitar mesmo estado várias vezes é penalizado
        4. Carregar antes da bateria baixa é penalizado
        5. Não carregar após bateria baixa é muito penalizado
        6. Não ir para trabalho é penalizado
        7. Não voltar para base é penalizado
        """
        if not hasattr(self, "cost_params"):
            self.cost_params = {}

        cp = self.cost_params

        # VALORES BALANCEADOS - Hierarquia clara de prioridades
        cp.setdefault("CUSTO_TEMPO_BASE", 0.1)              # Penalidade base por tempo
        cp.setdefault("PENALIDADE_PARADO", 0.5)             # Penalidade por ficar parado
        cp.setdefault("PENALIDADE_REPETICAO", 0.3)          # Penalidade por visitar estado repetido
        cp.setdefault("PENALIDADE_CARREGAMENTO_PREMATURO", 2.0)  # Carregar antes da bateria baixa
        cp.setdefault("PENALIDADE_NAO_CARREGAMENTO", 15.0)  # MUITO ALTA: Não carregar após bateria baixa
        cp.setdefault("PENALIDADE_NAO_TRABALHO", 3.0)       # Não ir para eventos de trabalho
        cp.setdefault("PENALIDADE_NAO_BASE", 2.0)           # Não voltar para base
        
        # INCENTIVOS (valores negativos) - CORREÇÃO: DEFINIR TODOS OS INCENTIVOS
        cp.setdefault("INCENTIVO_TAREFA_COMPLETA", -20.0)   # MAIOR INCENTIVO: Realizar tarefa completa
        cp.setdefault("INCENTIVO_COLETA", -8.0)             # Incentivo para coleta - AGORA DEFINIDO
        cp.setdefault("INCENTIVO_ENTREGA", -12.0)           # Incentivo para entrega - AGORA DEFINIDO
        cp.setdefault("INCENTIVO_CARREGAMENTO_CORRETO", -5.0) # Carregar quando necessário
        cp.setdefault("INCENTIVO_BASE", -2.0)               # Incentivo para estar na base

        CUSTO_TEMPO_BASE = cp["CUSTO_TEMPO_BASE"]
        PENALIDADE_PARADO = cp["PENALIDADE_PARADO"]
        INCENTIVO_TAREFA_COMPLETA = cp["INCENTIVO_TAREFA_COMPLETA"]
        INCENTIVO_COLETA = cp["INCENTIVO_COLETA"]  # DEFINIDO
        INCENTIVO_ENTREGA = cp["INCENTIVO_ENTREGA"]  # DEFINIDO
        INCENTIVO_BASE = cp["INCENTIVO_BASE"]

        self._precomputar_distancia_para_vertices_especiais()
        self.custos_estado_atomico.clear()

        # 1. CUSTOS BASE: Todos os estados têm custo de tempo
        for nome_automato, automato in self.Dicionario_Automatos.items():
            for estado in states(automato):
                estado_str = str(estado)
                # Penalidade base por tempo para todos os estados
                self.custos_estado_atomico[estado_str] = (0.0, 0.0, CUSTO_TEMPO_BASE)

        # 2. PENALIDADES POR OCIOSIDADE E REPETIÇÃO
        # Estados de movimento
        if "Parado" in self.custos_estado_atomico:
            self.custos_estado_atomico["Parado"] = (
                0.0,  # E
                0.0,  # Tf  
                CUSTO_TEMPO_BASE + PENALIDADE_PARADO  # D: Penalidade por parado
            )

        if "Movendo" in self.custos_estado_atomico:
            self.custos_estado_atomico["Movendo"] = (
                0.0,  # E (será calculado por aresta)
                0.0,  # Tf (será calculado por aresta)
                CUSTO_TEMPO_BASE  # D: Apenas custo base
            )

        # 3. CUSTOS DE MOVIMENTO BASEADOS EM DISTÂNCIA REAL
        for u, v, k, data in self.G.edges(keys=True, data=True):
            tempo_voo = self._obter_tempo_voo_aresta(u, v)
            consumo_energia = self._obter_consumo_energia_aresta(u, v)
            
            for estado_ocupado in [f"ocupado_{u}{v}", f"ocupado_{v}{u}"]:
                if estado_ocupado in self.custos_estado_atomico:
                    self.custos_estado_atomico[estado_ocupado] = (
                        consumo_energia,  # E: proporcional à distância
                        tempo_voo,        # Tf: tempo real de voo
                        CUSTO_TEMPO_BASE  # D: custo base
                    )

        # 4. INCENTIVOS E PENALIDADES POR LOCALIZAÇÃO
        for nome_no in self.G.nodes():
            estado_mapa = str(nome_no)
            if estado_mapa not in self.custos_estado_atomico: 
                continue

            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))
            custo_atual = self.custos_estado_atomico[estado_mapa]

            if tipo_no in {"ESTACAO", "VERTIPORT"}:
                # Base - incentivo moderado para voltar
                self.custos_estado_atomico[estado_mapa] = (
                    custo_atual[0],
                    custo_atual[1],
                    custo_atual[2] + INCENTIVO_BASE  # Incentivo para base
                )
            elif tipo_no in {"FORNECEDOR", "CLIENTE"}:
                # Locais de trabalho - incentivo para visitar
                self.custos_estado_atomico[estado_mapa] = (
                    custo_atual[0],
                    custo_atual[1],
                    custo_atual[2] - 1.0  # Pequeno incentivo
                )

        # 5. CUSTOS DE TRABALHO - MAIORES INCENTIVOS
        for nome_no in self.G.nodes():
            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))
            estado_trab = f"trabalhando_{nome_no}"

            if estado_trab in self.custos_estado_atomico:
                if tipo_no == "FORNECEDOR":
                    self.custos_estado_atomico[estado_trab] = (
                        0.0,  # E
                        0.0,  # Tf
                        INCENTIVO_COLETA  # D: Incentivo forte para coleta
                    )
                elif tipo_no == "CLIENTE":
                    self.custos_estado_atomico[estado_trab] = (
                        0.0,  # E
                        0.0,  # Tf  
                        INCENTIVO_ENTREGA  # D: Incentivo forte para entrega
                    )

        # Workflow global - MAIOR INCENTIVO
        if "pick" in self.custos_estado_atomico:
            self.custos_estado_atomico["pick"] = (0.0, 0.0, INCENTIVO_COLETA)
        if "place" in self.custos_estado_atomico:
            self.custos_estado_atomico["place"] = (0.0, 0.0, INCENTIVO_ENTREGA)

        # 6. CUSTOS DE BATERIA - PENALIDADES FORTES
        if "bat_baixa" in self.custos_estado_atomico:
            self.custos_estado_atomico["bat_baixa"] = (
                0.0,  # E
                0.0,  # Tf
                cp["PENALIDADE_NAO_CARREGAMENTO"]  # D: PENALIDADE MUITO ALTA
            )

        # Estados de carregamento
        for nome_no in self.G.nodes():
            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))
            if tipo_no == "ESTACAO":
                estado_carregando = f"carregando_{nome_no}"
                if estado_carregando in self.custos_estado_atomico:
                    self.custos_estado_atomico[estado_carregando] = (
                        0.0,  # E
                        0.0,  # Tf
                        cp["INCENTIVO_CARREGAMENTO_CORRETO"]  # D: Incentivo quando necessário
                    )

    def obter_custo_estado_supervisor(self, estado_supervisor) -> Tuple[float, float, float]:
        """
        Cálculo de custo final com fatores contextuais que atendam aos requisitos
        """
        cp = self.cost_params

        PENALIDADE_REPETICAO = cp["PENALIDADE_REPETICAO"]
        PENALIDADE_CARREGAMENTO_PREMATURO = cp["PENALIDADE_CARREGAMENTO_PREMATURO"]
        PENALIDADE_NAO_TRABALHO = cp["PENALIDADE_NAO_TRABALHO"]
        PENALIDADE_NAO_BASE = cp["PENALIDADE_NAO_BASE"]
        INCENTIVO_TAREFA_COMPLETA = cp["INCENTIVO_TAREFA_COMPLETA"]

        E_total, Tf_total, D_total = 0.0, 0.0, 0.0
        componentes = [c.strip() for c in str(estado_supervisor).split('|') if c.strip()]

        # 1) Soma dos custos base (70% do peso)
        estados_visitados = set()
        for estado_componente in componentes:
            if estado_componente in self.custos_estado_atomico:
                E, Tf, D = self.custos_estado_atomico[estado_componente]
                E_total += E
                Tf_total += Tf
                D_total += D
                
                # Penalidade por repetição de estados
                if estado_componente in estados_visitados:
                    D_total += PENALIDADE_REPETICAO
                estados_visitados.add(estado_componente)

        # 2) FATORES CONTEXTUAIS CRÍTICOS (30% do peso)
        map_node = next((c for c in componentes if c in self.G.nodes), None)
        is_moving = ("Movendo" in componentes)
        is_stopped = ("Parado" in componentes)
        is_low_battery = ("bat_baixa" in componentes)
        is_charging = any(c.startswith("carregando_") for c in componentes)
        is_working = any(c.startswith(("trabalhando_", "pick", "place")) for c in componentes)
        is_at_base = any(self._tipo_norm(self.G.nodes.get(c, {}).get("tipo", "")) 
                        in {"ESTACAO", "VERTIPORT"} for c in componentes if c in self.G.nodes)

        # 2.1) PENALIDADE: Carregar antes da bateria baixa
        if is_charging and not is_low_battery:
            D_total += PENALIDADE_CARREGAMENTO_PREMATURO

        # 2.2) PENALIDADE: Não carregar após bateria baixa (JÁ ESTÁ NO CUSTO BASE DO ESTADO)

        # 2.3) PENALIDADE: Não ir para trabalho quando possível
        if is_stopped and map_node and not is_working:
            tipo_no = self._tipo_norm(self.G.nodes[map_node].get("tipo", ""))
            if tipo_no in {"FORNECEDOR", "CLIENTE"}:
                D_total += PENALIDADE_NAO_TRABALHO

        # 2.4) PENALIDADE: Não voltar para base após tarefa completa
        if not is_at_base and not is_working and not is_moving:
            D_total += PENALIDADE_NAO_BASE

        # 2.5) INCENTIVO: Tarefa completa (maior incentivo)
        workflow_states = [c for c in componentes if c in ["pick", "place", "vantport"]]
        if len(workflow_states) >= 2:  # Completa pelo menos parte do workflow
            D_total += INCENTIVO_TAREFA_COMPLETA / len(workflow_states)

        # 3) FATORES SECUNDÁRIOS (balanceamento)
        if is_stopped and map_node:
            tipo_no = self._tipo_norm(self.G.nodes[map_node].get("tipo", ""))
            # Penalidade extra por parado em local não estratégico
            if tipo_no not in {"FORNECEDOR", "CLIENTE", "ESTACAO", "VERTIPORT"}:
                D_total += PENALIDADE_REPETICAO * 0.5

        return (E_total, Tf_total, D_total)

    def atualizar_parametros_custo(
        self,
        consumo_por_metro: float = None,
        velocidade_media: float = None,
        # Novos parâmetros para ajuste fino
        penalidade_parado: float = None,
        incentivo_tarefa: float = None,
        penalidade_nao_carregamento: float = None,
        penalidade_repeticao: float = None,
        incentivo_coleta: float = None,
        incentivo_entrega: float = None
    ):
        """
        Atualiza parâmetros de custo com novos valores
        """
        if not hasattr(self, "cost_params"):
            self.cost_params = {}

        updated = False

        # Parâmetros físicos
        if consumo_por_metro is not None:
            self.consumo_por_metro = float(abs(consumo_por_metro))
            updated = True

        if velocidade_media is not None:
            self.velocidade_media = float(abs(velocidade_media)) if velocidade_media != 0 else 2.0
            updated = True

        # Parâmetros de custo
        if penalidade_parado is not None:
            self.cost_params["PENALIDADE_PARADO"] = float(penalidade_parado)
            updated = True

        if incentivo_tarefa is not None:
            self.cost_params["INCENTIVO_TAREFA_COMPLETA"] = float(incentivo_tarefa)
            updated = True

        if penalidade_nao_carregamento is not None:
            self.cost_params["PENALIDADE_NAO_CARREGAMENTO"] = float(penalidade_nao_carregamento)
            updated = True

        if penalidade_repeticao is not None:
            self.cost_params["PENALIDADE_REPETICAO"] = float(penalidade_repeticao)
            updated = True

        # CORREÇÃO: Adicionar os novos parâmetros
        if incentivo_coleta is not None:
            self.cost_params["INCENTIVO_COLETA"] = float(incentivo_coleta)
            updated = True

        if incentivo_entrega is not None:
            self.cost_params["INCENTIVO_ENTREGA"] = float(incentivo_entrega)
            updated = True

        if updated:
            print("[INFO] Parâmetros de custo atualizados - recalculando custos...")
            if hasattr(self, "_dist_cache"):
                self._dist_cache.clear()
            self._inicializar_custos_estados()
            if hasattr(self, "supervisor_mono") and self.supervisor_mono is not None:
                self.criar_dicionario_custo_supervisor()

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
                    trs.append((s_base, self.ev(f"pega_{u}{n}"), s_base))

        
        A = dfa(trs, s_base, f"work_flow_{n}")
        self.Dicionario_Automatos[f"work_flow_{n}"] = A
        self.specs.append(A)

    def _automato_fim_de_carga(self):
        # Estados
        # 'Apto': O VANT está apto a carregar (está fora da estação, ou acabou de entrar/mover). Estado inicial.
        s_apto = state("apto_carregar", marked=True) 
        # 'Carregou': O VANT acabou de terminar o carregamento e precisa sair da estação.
        s_carregou = state("carregou_precisa_sair") 
        
        trs = []
        
        # Iterar sobre todos os nós para encontrar as Estações (ESTACAO)
        for n in self.G.nodes():
            tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
            if tipo == "ESTACAO":
                # Eventos de Início e Fim de Carregamento
                e_ini_carregar = self.ev(f"carregar_{n}")
                
                for x in self.G.neighbors(n): 
                    e_saida = self.ev(f"pega_{n}{x}")
                    if e_saida is not None:
                        trs.append((s_carregou, e_saida, s_apto))
                        trs.append((s_apto, e_saida, s_apto))
                    
                trs.append((s_apto, e_ini_carregar, s_carregou))
                


        # Criação do DFA (Deterministic Finite Automaton)
        A = dfa(trs, s_apto, "fim_de_carga")
        self.Dicionario_Automatos["fim_de_carga"] = A
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
        self._planning_horizon = 10         # Horizonte padrão para o MILP (ajuste se quiser)
        self._claimed_tasks = set()

        # SISTEMA DE BUFFER PIPELINE - NOVA LÓGICA COM MUTEX SEGURO
        self._execution_buffer = []           # Buffer de eventos controláveis calculados
        self._buffer_lock = threading.RLock()  # 🔥 MUDADO para RLock (permite reentrância)
        self._is_calculating_milp = False     # Flag para evitar cálculos concorrentes
        self._last_milp_calculation = 0       # Timestamp do último cálculo MILP
        self._min_calculation_interval = 0.1  # Intervalo mínimo entre cálculos MILP (segundos)

        self.dynamic_cost_dict = model.dicionario_custos_supervisor.copy()
        
        # Parâmetros de Penalidade por Persistência (ajustáveis)
        self.cost_params = {
        "PERSISTENCE_PENALTY_D": 0.1,      # Reduzido drasticamente
        "PERSISTENCE_PENALTY_TF": 0.02,    # Reduzido drasticamente  
        "CHARGE_INCENTIVE_MULT": 1.5,      # Moderado
        }
        
        # Rastreamento de tempo
        self.last_state_entry_time = time.time()
        
        # NOVO: Conjunto de eventos GENÉRICOS proibidos pelo UTM Central (atualizado via ROS)
        self._global_prohibited_generic = set()
        self._prohibited_lock = threading.Lock() # Para acesso seguro aos eventos proibidos

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

            self.terminou=[False,False,False]

            # Subscriber de tarefas (NOVO)
            self.sub_tarefas = rospy.Subscriber(
                "/task",
                String,
                self._callback_tarefas,
                queue_size=10
            )

            # Subscriber para eventos proibidos globais (do UTM Central)
            self.sub_global_prohibited=rospy.Subscriber(
                "/prohibited_events", 
                String,
                self._callback_eventos_proibidos, 
                queue_size=10
            )
                
            self.pub_tarefas_claim = rospy.Publisher("/task_claims", String, queue_size=10)
            self.sub_tarefas_claim = rospy.Subscriber(
                "/task_claims",
                String,
                self._callback_tarefas_claim,
                queue_size=10
            )
            rospy.sleep(0.3)
            self._publish_ros()

    # 🔥 MÉTODOS DE BUFFER COM MUTEX SEGURO E SEM DEADLOCKS
    def _get_buffer_size(self) -> int:
        """Retorna tamanho atual do buffer - COM MUTEX SEGURO"""
        with self._buffer_lock:
            return len(self._execution_buffer)

    def _clear_buffer(self):
        """Limpa o buffer - COM MUTEX SEGURO"""
        with self._buffer_lock:
            self._execution_buffer.clear()
            if self.enable_ros:
                import rospy
                rospy.loginfo(f"[{self.name}] Buffer limpo")

    def _add_to_buffer(self, event_name: str):
        """Adiciona evento ao buffer - COM MUTEX SEGURO"""
        with self._buffer_lock:
            # Substitui qualquer evento anterior no buffer (mantém apenas 1)
            self._execution_buffer = [event_name]
            
        if self.enable_ros:
            import rospy
            rospy.loginfo(f"[{self.name}] 🗂️ Evento armazenado no buffer: {event_name}")

    def _get_buffered_event(self) -> Optional[str]:
        """Recupera evento do buffer SEM REMOVER - COM MUTEX SEGURO"""
        with self._buffer_lock:
            if self._execution_buffer:
                return self._execution_buffer[0]
        return None

    def _remove_buffered_event(self):
        """Remove evento do buffer - COM MUTEX SEGURO"""
        with self._buffer_lock:
            if self._execution_buffer:
                self._execution_buffer.pop(0)

    def _is_globally_prohibited(self, ev_with_id: str) -> bool:
        """
        Verifica se o evento com sufixo _{id} está proibido pelo UTM.
        """
        m = self._RE_SUFFIX.match(ev_with_id)
        if not m:
            return False

        generic_name = m.group(1)
        with self._prohibited_lock:
            return generic_name in self._global_prohibited_generic
        
    def _update_dynamic_cost(self):
        """
        Atualização dinâmica balanceada - evita mudanças bruscas
        """
        if self.enable_ros:
            import rospy
            current_time = rospy.get_time()
        else:
            current_time = time.time()
            
        current_state_str = str(self._state)
        
        # 1. Sempre começa do custo base estático
        self.dynamic_cost_dict = self.model.dicionario_custos_supervisor.copy()
        
        # Parâmetros MUITO mais conservadores
        PENALTY_D = 0.1
        PENALTY_TF = 0.02
        CHARGE_MULT = 1.5

        # Persistência Leve
        time_spent = current_time - self.last_state_entry_time
        
        if time_spent > 2.0 and current_state_str in self.dynamic_cost_dict:
            E_base, Tf_base, D_base = self.dynamic_cost_dict[current_state_str]
            
            Tf_increase = min(time_spent * PENALTY_TF, 0.3)
            D_increase = min(time_spent * PENALTY_D, 0.5)
            
            self.dynamic_cost_dict[current_state_str] = (
                E_base,
                Tf_base + Tf_increase,
                D_base + D_increase
            )

        # Bateria Baixa Moderada
        if "bat_baixa" in current_state_str:
            base_incentive_E = self.model.cost_params.get("INCENTIVO_CARGA_E", -2.0)
            
            for state_str, (E, Tf, D) in self.dynamic_cost_dict.items():
                componentes = [c.strip() for c in state_str.split('|') if c.strip()]
                map_node = next((c for c in componentes if c in self.model.G.nodes), None)

                if map_node:
                    tipo_no = self.model._tipo_norm(self.model.G.nodes[map_node].get("tipo", ""))
                    
                    if tipo_no in {"ESTACAO", "VERTIPORT"}:
                        extra_incentive_E = base_incentive_E * (CHARGE_MULT - 1.0)
                        E_updated = E + extra_incentive_E
                        D_updated = D + (base_incentive_E * 0.5)
                        self.dynamic_cost_dict[state_str] = (E_updated, Tf, D_updated)

    def _on_event(self, msg):
        """Recebe eventos do barramento /event (String)."""
        ev = str(msg.data or "")
        if ev == "ping":
            self._publish_ros()
            return
        _ = self.step(ev)

    # ----------------------- CALLBACK DE TAREFAS (NOVO) -----------------------

    def _callback_eventos_proibidos(self, msg):
        """
        Recebe a string de eventos genéricos proibidos globalmente pelo UTM Central
        e atualiza o conjunto interno.
        """
        raw = str(msg.data or "").strip()
        with self._prohibited_lock:
            if not raw:
                self._global_prohibited_generic = set()
            else:
                self._global_prohibited_generic = set(raw.split(','))
        
    def _callback_tarefas_claim(self, msg):
        """
        Recebe claims de tarefas no formato ID_TAREFA:FORNECEDOR_X,CLIENTE_Y
        e registra que essa tarefa já foi pega por algum VANT.
        """
        raw = str(msg.data or "").strip()
        if not raw:
            return
        self._claimed_tasks.add(raw)

    def _callback_tarefas(self, msg):
        """
        Recebe tarefas no formato ATUALIZADO:
            'ID_TAREFA:FORNECEDOR_X,CLIENTE_Y'
        """
        if not self.enable_ros:
            return
            
        import rospy
        from std_msgs.msg import String
        import random

        raw = str(msg.data or "").strip()
        if not raw:
            return

        # PARSING
        if ":" not in raw:
            rospy.logwarn(f"[{self.name}] Formato inválido de tarefa recebida: '{raw}'. Esperado 'ID:FORNECEDOR,CLIENTE'.")
            self.pub_cmd_event.publish(String(data=f"rejeita_tarefa_{self.id}" ))
            return
            
        try:
            task_id, nodes_raw = raw.split(":", 1)
        except ValueError:
            rospy.logwarn(f"[{self.name}] Erro no parsing do ID da tarefa: '{raw}'.")
            self.pub_cmd_event.publish(String(data=f"rejeita_tarefa_{self.id}" ))
            return

        parts = [p.strip() for p in nodes_raw.split(",") if p.strip()]
        if len(parts) != 2:
            rospy.logwarn(f"[{self.name}] Formato inválido de nós na tarefa: '{nodes_raw}'. Esperado 'FORNECEDOR,CLIENTE'.")
            self.pub_cmd_event.publish(String(data=f"rejeita_tarefa_{self.id}" ))
            return

        fornecedor, cliente = parts[0], parts[1]

        # 1. Verifica se tarefa já foi claimada ou se VANT está ocupado
        if raw in self._claimed_tasks:
            rospy.loginfo(f"[{self.name}] Tarefa '{raw}' já foi claimada. Ignorando.")
            self.pub_cmd_event.publish(String(data=f"rejeita_tarefa_{self.id}" ))
            return

        if self._tarefa_ativa is not None:
            rospy.loginfo(f"[{self.name}] Já possuo tarefa ativa {self._tarefa_ativa}. Ignorando '{raw}'.")
            self.pub_cmd_event.publish(String(data=f"rejeita_tarefa_{self.id}" ))
            return

        # 2. Pequeno atraso aleatório para evitar colisão
        d = 0.5
        ai0 = (self.id) * (d + 1)
        aik = ai0 + d
        delay = random.uniform(ai0, aik)
        rospy.loginfo(f"[{self.name}] Atraso de {delay:.2f}s para evitar colisão de CLAIM na tarefa '{raw}'.")
        rospy.sleep(delay)

        # 3. Verificação final após delay
        if raw in self._claimed_tasks:
            rospy.loginfo(f"[{self.name}] Após delay, tarefa '{raw}' já foi claimada por outro VANT. Ignorando.")
            self.pub_cmd_event.publish(String(data=f"rejeita_tarefa_{self.id}" ))
            return

        # 4. SUCESSO! Publica CLAIM e ativa tarefa
        rospy.loginfo(f"[{self.name}] VENCEDOR: Publicando CLAIM para tarefa: {raw}.")
        self.pub_tarefas_claim.publish(String(data=raw))
        self._tarefa_ativa = raw

        rospy.loginfo(f"[{self.name}] Tarefa recebida e **CLAIMADA** com sucesso: {self._tarefa_ativa}.")
        self.pub_cmd_event.publish(String(data=f"aceita_tarefa_{self.id}" ))

        # 🚀 INÍCIO DO PIPELINE
        rospy.loginfo(f"[{self.name}] 🚀 Iniciando pipeline sequencial para tarefa: {self._tarefa_ativa}")
        self._trigger_milp_calculation()

    # ----------------------- API LÓGICA PARA TESTE -----------------------
    def state(self):
        return self._state

    def enabled_events(self):
        """Eventos habilitados (já com sufixo _{id}) como strings, FILTRADOS pelas proibições globais do UTM."""
        s = str(self._state)
        factible_events_id = set()
        
        for (q, e, _d) in self._trs_id:
            if str(q) == s:
                factible_events_id.add(str(e))
        
        enabled_events_id = set()
        with self._prohibited_lock:
            global_prohibited = self._global_prohibited_generic.copy()
            
        for ev_id in factible_events_id:
            generic_name = self.rev_event_map.get(ev_id)
            if generic_name is None or generic_name not in global_prohibited:
                enabled_events_id.add(ev_id)
        
        return sorted(list(enabled_events_id))

    def _should_process(self, ev: str) -> bool:
        """Verifica se o evento deve ser processado por este VANT"""
        m = self._RE_SUFFIX.match(ev)
        if not m:
            return False
        return (int(m.group(2)) == self.id)

    def _trigger_milp_calculation(self):
        """Dispara cálculo MILP apenas se não estiver calculando - COM LOCK SEGURO"""
        with self._milp_thread_lock:
            if not self._is_calculating_milp and (self._milp_thread is None or not self._milp_thread.is_alive()):
                self._milp_thread = threading.Thread(
                    target=self._run_milp_for_current_task,
                    daemon=True
                )
                self._milp_thread.start()

    def _run_milp_for_current_task(self):
        """
        CORREÇÃO: Priorização de VERTIPORT na fase de retorno
        """
        if not self.enable_ros:
            return

        import rospy
        
        # Marcar que está calculando
        self._is_calculating_milp = True
        self._last_milp_calculation = time.time()
        
        try:
            # 🚨 VERIFICAÇÃO ROBUSTA DE CONCLUSÃO DE TAREFA
            if self._tarefa_ativa is None:
                rospy.loginfo(f"[{self.name}] Tentativa de MILP sem tarefa ativa. Abortando.")
                return

            # ✅ VERIFICAÇÃO COMPLETA DA TAREFA - COM ATUALIZAÇÃO
            is_task_completely_finished = (
                self.terminou[0] and  # Coleta concluída
                self.terminou[1] and  # Entrega concluída  
                self.terminou[2]      # Retorno à base concluído
            )
            
            if is_task_completely_finished:
                rospy.loginfo(f"[{self.name}] 🎉 TAREFA COMPLETAMENTE CONCLUÍDA - Parando cálculo MILP")
                self._complete_current_task()
                return

            tarefa_completa = self._tarefa_ativa
            task_id, nodes_raw = tarefa_completa.split(":", 1)
            fornecedor, cliente = nodes_raw.split(",")

            # Cálculo do MILP
            H = self._planning_horizon
            
            # 🎯 EVENTOS DE INTERESSE DINÂMICOS - PRIORIZANDO VERTIPORT
            eventos_interesse_gen = []
            volta = []
            
            if self.terminou[0] and self.terminou[1]: 
                # Fase 3: Voltando para VERTIPORT (prioridade máxima)
                if not self.terminou[2]:
                    # 🏁 PRIORIDADE: VERTIPORT primeiro, depois ESTACAO
                    vertiports = []
                    estacoes = []
                    
                    for n in self.model.G.nodes():
                        tipo = self.model._tipo_norm(self.model.G.nodes[n].get("tipo", ""))
                        if tipo == "VERTIPORT":
                            vertiports.append(n)
                        elif tipo == "ESTACAO":
                            estacoes.append(n)
                    
                    # Primeiro tenta VERTIPORTS
                    for n in vertiports:
                        for u, v, k, data in self.model.G.in_edges(n, keys=True, data=True):
                            eventos_interesse_gen.append(f"pega_{u}{n}")
                            volta.append(f"pega_{u}{n}_{self.id}")
                    
                    # Se não encontrou VERTIPORTS, tenta ESTAÇÕES
                    if not eventos_interesse_gen:
                        for n in estacoes:
                            for u, v, k, data in self.model.G.in_edges(n, keys=True, data=True):
                                eventos_interesse_gen.append(f"pega_{u}{n}")
                                volta.append(f"pega_{u}{n}_{self.id}")
                    
                    if eventos_interesse_gen:
                        rospy.loginfo(f"[{self.name}] 🏠 Fase 3: Retornando à base - {len(vertiports)} VERTIPORT(s), {len(estacoes)} ESTACAO(s) encontrados")
                    else:
                        rospy.logwarn(f"[{self.name}] ❌ Nenhuma base (VERTIPORT/ESTACAO) encontrada para retorno!")
            else:
                # Fase 1 e 2: Trabalho no fornecedor e cliente
                # ✅ APENAS eventos de interesse que AINDA NÃO ACONTECERAM
                if not self.terminou[0]:  # Se coleta NÃO aconteceu
                    eventos_interesse_gen.append(f"comeca_trabalho_{fornecedor}")
                    rospy.loginfo(f"[{self.name}] 📦 Fase 1: Coleta pendente")
                
                if not self.terminou[1]:  # Se entrega NÃO aconteceu  
                    eventos_interesse_gen.append(f"comeca_trabalho_{cliente}")
                    rospy.loginfo(f"[{self.name}] 🚚 Fase 2: Entrega pendente")
            
            # ✅ SE TODAS AS FASES FORAM CONCLUÍDAS, PARA O MILP
            if not eventos_interesse_gen and not self.terminou[2]:
                rospy.logwarn(f"[{self.name}] ⚠️ Nenhum evento de interesse - tarefa pode estar concluída")
                # Força verificação final
                if self.terminou[0] and self.terminou[1] and not self.terminou[2]:
                    rospy.loginfo(f"[{self.name}] 🔍 Verificando se retorno à base foi concluído...")
                    # Força detecção de base pelo estado atual
                    self._detect_base_return("", "")
                else:
                    self._complete_current_task()
                    return

            # Converter para eventos com ID
            eventos_interesse_id = [f"{nm}_{self.id}" for nm in eventos_interesse_gen]
            
            rospy.loginfo(f"[{self.name}] 🎯 Eventos de interesse DINÂMICOS: {eventos_interesse_id}")
            rospy.loginfo(f"[{self.name}] 📊 Progresso: coleta={self.terminou[0]}, entrega={self.terminou[1]}, base={self.terminou[2]}")

            with self._prohibited_lock:
                global_prohibited_generic = self._global_prohibited_generic.copy()

            eventos_proibidos_id = []
            for ev_gen in global_prohibited_generic:
                ev_id = self.event_map.get(ev_gen)
                if ev_id:
                    eventos_proibidos_id.append(ev_id)
            
            estado_inicial = self._state
            cost_dict = self.dynamic_cost_dict

            rospy.loginfo(f"[{self.name}] 🔄 Calculando PRÓXIMO evento controlável via MILP (H={H})")

            # Chamada ao otimizador
            event_seq, status = otimizador(
                self.supervisor,
                estado_inicial,
                H,
                cost_dict,
                eventos_interesse_id,
                eventos_proibidos_id
            )

            rospy.loginfo(f"[{self.name}] MILP retornou status={status}, seq={event_seq}")

            # 🚨 VERIFICAÇÃO FINAL ANTES DE PROCESSAR RESULTADO
            if self._tarefa_ativa is None:
                rospy.loginfo(f"[{self.name}] Tarefa foi cancelada durante cálculo MILP")
                return

            # Busca APENAS o PRIMEIRO evento controlável da sequência
            next_controllable = None
            
            for ev_name in event_seq:
                ev_obj = self._event_objects.get(ev_name)
                
                if ev_obj is None:
                    continue
                    
                if is_controllable(ev_obj):
                    next_controllable = ev_name
                    
                    # Atualização do progresso (apenas para logging)
                    if next_controllable == f"comeca_trabalho_{fornecedor}_{self.id}":
                        rospy.loginfo(f"[{self.name}] 🎯 Evento de COLETA encontrado na sequência MILP")
                    elif next_controllable == f"comeca_trabalho_{cliente}_{self.id}":
                        rospy.loginfo(f"[{self.name}] 🎯 Evento de ENTREGA encontrado na sequência MILP")
                    # 🎯 CORREÇÃO: Detecção de evento de retorno à base sem variável 'u' indefinida
                    elif any(base_node in next_controllable for base_node in ["VERTIPORT", "ESTACAO"]):
                        rospy.loginfo(f"[{self.name}] 🎯 Evento de RETORNO À BASE encontrado na sequência MILP: {next_controllable}")
                    
                    rospy.loginfo(f"[{self.name}] ✅ Próximo evento controlável encontrado: {next_controllable}")
                    break

            # 🔥 ARMAZENA NO BUFFER COM MUTEX SEGURO
            if next_controllable is not None:
                self._add_to_buffer(next_controllable)
                
                # ✅ Tenta publicar IMEDIATAMENTE (fora do mutex principal)
                self._publish_buffered_event_if_enabled()
            else:
                rospy.logwarn(f"[{self.name}] ❌ Nenhum evento controlável encontrado na sequência MILP")
                # Se não encontrou evento controlável, verifica se tarefa está concluída
                if self.terminou[0] and self.terminou[1] and self.terminou[2]:
                    rospy.loginfo(f"[{self.name}] 🎉 Tarefa completamente concluída - limpando buffer")
                    self._complete_current_task()
                else:
                    rospy.loginfo(f"[{self.name}] ⏳ Aguardando eventos não controláveis para progresso...")
                    self._clear_buffer()

        except Exception as e:
            rospy.logerr(f"[{self.name}] Erro executando MILP: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self._clear_buffer()
        finally:
            self._is_calculating_milp = False

    def _complete_current_task(self):
        """Limpa COMPLETAMENTE o estado do VANT após conclusão da missão"""
        if self._tarefa_ativa is not None:
            import rospy
            rospy.loginfo(f"[{self.name}] 🎉 MISSÃO COMPLETAMENTE CONCLUÍDA: {self._tarefa_ativa}")
            rospy.loginfo(f"[{self.name}] 📊 Estatísticas finais: coleta={self.terminou[0]}, entrega={self.terminou[1]}, base={self.terminou[2]}")
        
        # 🚨 LIMPEZA COMPLETA E SEGURA
        with self._milp_thread_lock:
            self._tarefa_ativa = None
            self.terminou = [False, False, False]  # Reset para próxima tarefa
            self._is_calculating_milp = False
            
        self._clear_buffer()
        
        # Publica evento de conclusão se ROS estiver ativo
        if self.enable_ros:
            import rospy
            from std_msgs.msg import String
            rospy.loginfo(f"[{self.name}] 📢 Publicando conclusão de tarefa")
            self.pub_cmd_event.publish(String(data=f"completa_tarefa_{self.id}"))

    def step(self, ev: str) -> bool:
        """
        CORREÇÃO: Verificação robusta de conclusão de tarefa com atualização automática
        """
        if not self._should_process(ev):
            return False
        
        # 1) UTM: bloqueio global
        if self._is_globally_prohibited(ev):
            if self.enable_ros:
                import rospy
                rospy.logwarn(f"[{self.name}] Evento '{ev}' bloqueado globalmente pelo UTM. Ignorando.")
            return False
        
        # Converter string para objeto Event correspondente
        event_obj = self._event_objects.get(ev)
        if event_obj is None:
            return False

        s = str(self._state)
        transicionou = False

        # Processa transição de estado
        for (q, e, d) in self._trs_id:
            if str(q) == s and e == event_obj:
                self._state = d
                transicionou = True
                
                if self.enable_ros:
                    import rospy
                    self.last_state_entry_time = rospy.get_time()
                    self._publish_ros()
                else:
                    self.last_state_entry_time = time.time()
                break

        if not transicionou:
            return False

        # 🚨 ATUALIZAÇÃO AUTOMÁTICA DO PROGRESSO DA TAREFA
        self._update_task_progress(ev)

        # Atualização de custos dinâmicos
        self._update_dynamic_cost()
        
        # ✅ VERIFICAÇÃO EXTRA: Força detecção de base se necessário
        if self.terminou[0] and self.terminou[1] and not self.terminou[2]:
            self._detect_base_return("", "")  # Força verificação pelo estado atual
        
        # ✅ VERIFICAÇÃO ROBUSTA DE CONCLUSÃO
        is_task_completely_finished = (
            self.terminou[0] and  # Coleta concluída
            self.terminou[1] and  # Entrega concluída  
            self.terminou[2]      # Retorno à base concluído
        )
        
        if self._tarefa_ativa is not None and is_task_completely_finished:
            if self.enable_ros:
                import rospy
                rospy.loginfo(f"[{self.name}] 🎉 TAREFA DETECTADA COMO CONCLUÍDA NO STEP - Finalizando")
            self._complete_current_task()
            return True

        # Continua com a lógica normal apenas se a tarefa não estiver concluída
        if self._tarefa_ativa is not None and not is_task_completely_finished:
            evento_foi_nao_controlavel = not is_controllable(event_obj)
            evento_foi_controlavel = is_controllable(event_obj)
            
            # 🔥 LÓGICA PRINCIPAL COM MUTEX SEGURO
            if evento_foi_controlavel:
                # ✅ Publicou evento controlável → Calcula próximo
                if self.enable_ros:
                    import rospy
                    rospy.loginfo(f"[{self.name}] 🔄 Gatilho: evento CONTROLÁVEL '{ev}' publicado → Calculando próximo MILP")
                self._trigger_milp_calculation()
                
            elif evento_foi_nao_controlavel:
                # 📨 Recebeu evento NÃO controlável → Publica próximo do buffer
                if self.enable_ros:
                    import rospy
                    rospy.loginfo(f"[{self.name}] 🔄 Gatilho: evento NÃO controlável '{ev}' recebido → Publicando próximo do buffer")
                self._publish_buffered_event_if_enabled()

        return True


    def _update_task_progress(self, ev: str):
        """
        ATUALIZAÇÃO ROBUSTA: Atualiza automaticamente o progresso da tarefa baseado nos eventos
        CORREÇÃO: Detecção correta de retorno ao VERTIPORT
        """
        if not self._tarefa_ativa:
            return
            
        if self.enable_ros:
            import rospy
            
        # Extrai informações da tarefa atual
        try:
            tarefa_completa = self._tarefa_ativa
            task_id, nodes_raw = tarefa_completa.split(":", 1)
            fornecedor, cliente = nodes_raw.split(",")
        except:
            return

        evento_generico = self.to_generic(ev)
        
        # 🎯 DETECÇÃO AUTOMÁTICA DE CONCLUSÃO DE FASE
        if evento_generico == f"fim_trabalho_{fornecedor}":
            if not self.terminou[0]:
                self.terminou[0] = True
                if self.enable_ros:
                    rospy.loginfo(f"[{self.name}] ✅ FASE 1 CONCLUÍDA: Coleta finalizada em {fornecedor}")
                    
        elif evento_generico == f"fim_trabalho_{cliente}":
            if not self.terminou[1]:
                self.terminou[1] = True
                if self.enable_ros:
                    rospy.loginfo(f"[{self.name}] ✅ FASE 2 CONCLUÍDA: Entrega finalizada em {cliente}")
        
        # 🏠 DETECÇÃO DE RETORNO À BASE (VERTIPORT) - CORRIGIDA
        self._detect_base_return(ev, evento_generico)
        
        # 📊 LOG DE PROGRESSO ATUALIZADO
        if self.enable_ros and any(self.terminou):
            rospy.loginfo(f"[{self.name}] 📊 PROGRESSO ATUALIZADO: coleta={self.terminou[0]}, entrega={self.terminou[1]}, base={self.terminou[2]}")

    def _detect_base_return(self, ev: str, evento_generico: str):
        """
        CORREÇÃO: Detecção específica e robusta de retorno ao VERTIPORT
        """
        if self.enable_ros:
            import rospy
        
        # Só detecta retorno se as fases 1 e 2 estiverem concluídas
        if not (self.terminou[0] and self.terminou[1]) or self.terminou[2]:
            return
        
        # 🔍 MÉTODO 1: Detecção por evento de chegada ao VERTIPORT
        if evento_generico.startswith("chega_") or evento_generico.startswith("libera_"):
            # Extrai o nó de destino do evento
            if evento_generico.startswith("chega_"):
                # Formato: chega_X_Y → Y é o destino
                partes = evento_generico.split('_')
                if len(partes) >= 3:
                    no_destino = partes[2]  # Segundo elemento após 'chega'
            else:  # libera_
                # Formato: libera_X_Y → Y é o destino  
                partes = evento_generico.split('_')
                if len(partes) >= 3:
                    no_destino = partes[2]  # Segundo elemento após 'libera'
            
            # Verifica se o nó de destino é um VERTIPORT
            if hasattr(self, 'model') and no_destino in self.model.G.nodes:
                tipo_no = self.model._tipo_norm(self.model.G.nodes[no_destino].get("tipo", ""))
                if tipo_no == "VERTIPORT":
                    if not self.terminou[2]:
                        self.terminou[2] = True
                        if self.enable_ros:
                            rospy.loginfo(f"[{self.name}] ✅ FASE 3 CONCLUÍDA: Retorno ao VERTIPORT {no_destino} detectado via evento")
                        return
        
        # 🔍 MÉTODO 2: Detecção por análise do estado atual do supervisor
        estado_atual = str(self._state)
        componentes = [c.strip() for c in estado_atual.split('|') if c.strip()]
        
        for componente in componentes:
            if componente in self.model.G.nodes:
                tipo_no = self.model._tipo_norm(self.model.G.nodes[componente].get("tipo", ""))
                if tipo_no == "VERTIPORT":
                    if not self.terminou[2]:
                        self.terminou[2] = True
                        if self.enable_ros:
                            rospy.loginfo(f"[{self.name}] ✅ FASE 3 CONCLUÍDA: Estado atual no VERTIPORT {componente}")
                        return
        
        # 🔍 MÉTODO 3: Detecção por eventos de carregamento no VERTIPORT
        if evento_generico.startswith("carregar_"):
            partes = evento_generico.split('_')
            if len(partes) >= 2:
                no_carregamento = partes[1]
                if no_carregamento in self.model.G.nodes:
                    tipo_no = self.model._tipo_norm(self.model.G.nodes[no_carregamento].get("tipo", ""))
                    if tipo_no == "VERTIPORT":
                        if not self.terminou[2]:
                            self.terminou[2] = True
                            if self.enable_ros:
                                rospy.loginfo(f"[{self.name}] ✅ FASE 3 CONCLUÍDA: Carregamento no VERTIPORT {no_carregamento}")
                        return
    def _publish_buffered_event_if_enabled(self):
            """Tenta publicar o evento do buffer se estiver habilitado - COM MUTEX SEGURO"""
            import rospy
            from std_msgs.msg import String
            
            # 🔥 PRIMEIRO: Verifica se há evento no buffer (com mutex rápido)
            buffered_event = self._get_buffered_event()
            if not buffered_event:
                return False
            
            # 🔥 SEGUNDO: Verifica se está habilitado (sem mutex para evitar deadlock)
            enabled_events = self.enabled_events()
            
            if buffered_event in enabled_events:
                rospy.loginfo(f"[{self.name}] 🚀 Publicando evento do buffer: {buffered_event}")
                self.pub_cmd_event.publish(String(data=buffered_event))
                
                # 🔥 TERCEIRO: Remove do buffer (com mutex rápido)
                self._remove_buffered_event()
                return True
            else:
                rospy.loginfo(f"[{self.name}] ⏳ Evento do buffer não habilitado: {buffered_event}")
                return False

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

    def to_generic(self, ev_with_id: str) -> str:
        return self.rev_event_map.get(ev_with_id, ev_with_id)

    def run(self):
        """Loop principal (apenas quando ROS está habilitado)."""
        if not self.enable_ros:
            return
        import rospy
        rospy.spin()




