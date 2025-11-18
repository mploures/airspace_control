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
    def _calcular_distancia_entre_nos(self, no1: str, no2: str) -> float:
        """Calcula a distância real entre dois nós baseado nas posições do stage"""
        if no1 in self.posicoes and no2 in self.posicoes:
            _, (x1, y1) = self.posicoes[no1]
            _, (x2, y2) = self.posicoes[no2]
            return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        return 1.0  # distância padrão se não encontrar posições

    def _obter_tempo_voo_aresta(self, u: str, v: str) -> float:
        """Calcula tempo de voo baseado na distância real entre nós"""
        distancia = self._calcular_distancia_entre_nos(u, v)
        velocidade_media = 2.0  # m/s - ajuste conforme seu sistema
        return distancia / velocidade_media

    def _obter_consumo_energia_aresta(self, u: str, v: str) -> float:
        """Calcula consumo de energia baseado na distância real - RETORNA CUSTO POSITIVO"""
        distancia = self._calcular_distancia_entre_nos(u, v)
        consumo_por_metro = 0.1  # ajuste conforme seu sistema
        return distancia * consumo_por_metro  # POSITIVO pois é custo

    def _inicializar_custos_estados(self):
        """
        Inicializa custos W = [E, Tf, D] com base em uma filosofia de "custo de oportunidade".
        
        Dimensões:
        - E (Energia):   Positivo = Custo; Negativo = Incentivo (Carregar)
        - Tf (Tempo):    Positivo = Custo (Duração)
        - D (Progresso): Positivo = Custo (Penalidade de Tempo); Negativo = Incentivo (Missão)
        """
        
        # ==================================================================
        # 1. CONSTANTES DE CUSTO (Ajuste estes valores para calibrar)
        # ==================================================================
        
        # (D) Penalidade base por passo de tempo. Isso torna a inatividade custosa.
        CUSTO_TEMPO_D = 0.5 
        
        # (E) Custo de energia para movimento (além do custo da aresta)
        CUSTO_MOVIMENTO_E = 0.2
        
        # (E) Custo de energia para operar (hovering em nós de trabalho)
        CUSTO_OPERACIONAL_E = 0.1
        
        # (E) Incentivo (negativo) por estar em um nó de carregamento
        INCENTIVO_CARGA_E = -1.0
        
        # (D) Incentivo (negativo) para progresso de missão
        INCENTIVO_COLETA_D = -5.0
        INCENTIVO_ENTREGA_D = -10.0
        
        # (E, D) Penalidades severas para estados indesejados
        PENALIDADE_BATERIA_E = 10.0
        PENALIDADE_BATERIA_D = 10.0

        
        # ==================================================================
        # 2. INICIALIZAÇÃO: CUSTO DE OPORTUNIDADE (D)
        # ==================================================================
        # Todos os estados atômicos começam com um custo de progresso positivo (penalidade de tempo).
        # Isso corrige o problema do "custo zero" para estados ociosos.
        for nome_automato, automato in self.Dicionario_Automatos.items():
            for estado in states(automato):
                self.custos_estado_atomico[str(estado)] = (
                    0.0,            # E (Energia)
                    0.0,            # Tf (Tempo Físico)
                    CUSTO_TEMPO_D   # D (Progresso) - Penalidade de tempo
                )
        
        # ==================================================================
        # 3. CUSTOS DE MOVIMENTO (E, Tf)
        # ==================================================================
        
        # Custo genérico de estar no estado "Movendo" (do _automato_movimento)
        if "Movendo" in self.custos_estado_atomico:
            self.custos_estado_atomico["Movendo"] = (
                CUSTO_MOVIMENTO_E,  # E: Custo base de energia para se mover
                0.1,                # Tf: Custo base de tempo
                CUSTO_TEMPO_D       # D: Mantém a penalidade de tempo
            )
            
        # Custo específico da aresta (do _automatos_arestas)
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            if chave not in self.dict_aresta_eventos:
                continue # Garante que o evento foi criado
                
            tempo_voo = self._obter_tempo_voo_aresta(u, v)
            consumo_energia = self._obter_consumo_energia_aresta(u, v)
            
            for estado_ocupado in [f"ocupado_{u}{v}", f"ocupado_{v}{u}"]:
                if estado_ocupado in self.custos_estado_atomico:
                    # Este custo é SOMADO ao custo de "Movendo"
                    self.custos_estado_atomico[estado_ocupado] = (
                        consumo_energia,  # E: Custo (gasto)
                        tempo_voo,        # Tf: Custo (duração)
                        CUSTO_TEMPO_D     # D: Mantém a penalidade de tempo
                    )
        
        # ==================================================================
        # 4. CUSTOS/INCENTIVOS DE LOCALIZAÇÃO (E)
        # ==================================================================
        # ATENÇÃO: Usamos os estados do _automato_mapa (ex: "VERTIPORT_0")
        # em vez de "dentro_{n}", pois os logs mostraram que o VANT
        # pode estar no nó (mapa) mas "fora" (loc_{n}) no estado inicial.
        
        for nome_no in self.G.nodes():
            estado_mapa = str(nome_no) # Estado do _automato_mapa
            if estado_mapa not in self.custos_estado_atomico:
                continue
                
            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))
            
            if tipo_no in {"ESTACAO", "VERTIPORT"}:
                # INCENTIVO de energia por estar em local de carga
                self.custos_estado_atomico[estado_mapa] = (
                    INCENTIVO_CARGA_E,  # E: Incentivo (negativo)
                    0.0,                # Tf
                    CUSTO_TEMPO_D       # D: Mantém a penalidade de tempo
                )
            elif tipo_no in {"FORNECEDOR", "CLIENTE"}:
                # CUSTO de energia por estar em local de trabalho (hovering)
                self.custos_estado_atomico[estado_mapa] = (
                    CUSTO_OPERACIONAL_E, # E: Custo (positivo)
                    0.0,                 # Tf
                    CUSTO_TEMPO_D        # D: Mantém a penalidade de tempo
                )
            # Nós lógicos (sem tipo) mantêm o custo (0.0 E, 0.0 Tf, 0.5 D)

        # ==================================================================
        # 5. INCENTIVOS DE PROGRESSO DE MISSÃO (D)
        # ==================================================================
        
        # Estados de trabalho ativo (do _automato_modos)
        for nome_no in self.G.nodes():
            tipo_no = self._tipo_norm(self.G.nodes[nome_no].get("tipo", ""))
            estado_trabalhando = f"trabalhando_{nome_no}"
            
            if estado_trabalhando in self.custos_estado_atomico:
                if tipo_no == "FORNECEDOR":
                    # GRANDE INCENTIVO de progresso (D negativo)
                    self.custos_estado_atomico[estado_trabalhando] = (
                        CUSTO_OPERACIONAL_E,  # E: Custo de operar
                        0.0,                  # Tf
                        INCENTIVO_COLETA_D    # D: Incentivo (negativo)
                    )
                elif tipo_no == "CLIENTE":
                    # INCENTIVO MÁXIMO de progresso (D negativo)
                    self.custos_estado_atomico[estado_trabalhando] = (
                        CUSTO_OPERACIONAL_E,  # E: Custo de operar
                        0.0,                  # Tf
                        INCENTIVO_ENTREGA_D   # D: Incentivo (negativo)
                    )

        # Estados do Workflow (do _automato_trabalho)
        if "pick" in self.custos_estado_atomico:
            self.custos_estado_atomico["pick"] = (
                0.0,                # E
                0.0,                # Tf
                INCENTIVO_COLETA_D  # D: Incentivo (negativo)
            )
        if "place" in self.custos_estado_atomico:
            self.custos_estado_atomico["place"] = (
                0.0,                 # E
                0.0,                 # Tf
                INCENTIVO_ENTREGA_D  # D: Incentivo (negativo)
            )

        # ==================================================================
        # 6. PENALIDADES (E, D)
        # ==================================================================
        
        # Estado de bateria baixa (do _automato_bateria_movimento)
        estado_bat_baixa = "bat_baixa"
        if estado_bat_baixa in self.custos_estado_atomico:
            # CUSTO MUITO ALTO (penalidade) em E e D
            self.custos_estado_atomico[estado_bat_baixa] = (
                PENALIDADE_BATERIA_E,  # E: Penalidade
                0.0,                   # Tf
                PENALIDADE_BATERIA_D   # D: Penalidade
            )


    def obter_custo_estado_supervisor(self, estado_supervisor) -> Tuple[float, float, float]:
        """Calcula custo W = [E, Tf, D] para um estado do supervisor somando estados componentes"""
        E_total, Tf_total, D_total = 0.0, 0.0, 0.0
        
        # O estado do supervisor é uma tupla de estados atômicos
        for estado_componente in estado_supervisor.split('|'):
            estado_str = str(estado_componente)
            if estado_str in self.custos_estado_atomico:
                E, Tf, D = self.custos_estado_atomico[estado_str]
                E_total += E
                Tf_total += Tf
                D_total += D
        
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
        
            custo_combinado = self.obter_custo_estado_supervisor(str(estado_supervisor))
                    
            custos_supervisor[str(estado_supervisor)] = custo_combinado
                    
        print(f"[INFO] Dicionário de custos criado para {len(custos_supervisor)} estados do supervisor.")
        
        return custos_supervisor
        
    def atualizar_parametros_custo(self, 
                                   consumo_por_metro: float = None,
                                   velocidade_media: float = None,
                                   ganho_carregamento: float = None):
        """Atualiza parâmetros e recalcula custos"""
        # Aqui você pode adicionar lógica para atualizar parâmetros
        # e chamar _inicializar_custos_estados() novamente se necessário
        if any(param is not None for param in [consumo_por_metro, velocidade_media, ganho_carregamento]):
            print("[INFO] Parâmetros de custo atualizados - recalculando...")
            self._inicializar_custos_estados()

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

        # 1) Recupera (ou calcula) supervisor genérico
        if supervisor_mono is None:
            if getattr(model, "supervisor_mono", None) is None:
                # Chama a função de cálculo se necessário
                supervisor_mono = model.compute_monolithic_supervisor() 
            else:
                supervisor_mono = model.supervisor_mono
        self._sup_gen = supervisor_mono

        # 2) Renomeia TODAS as transições do supervisor para evento_{id}
        trs_gen = list(transitions(self._sup_gen))
        renamed_trs = []
        self.event_map = {}      # e_gen -> e_id (string)
        self.rev_event_map = {}  # e_id (string) -> e_gen (AbstractEvent)
        self._event_objects = {} # e_id (string) -> AbstractEvent object
        
        for (q, e, q2) in trs_gen:
            es = str(e)
            es_id = f"{es}_{self.id}"
            
            # Criar novo objeto Event com o nome modificado, mantendo a controlabilidade
            new_event = event(es_id, controllable=is_controllable(e))
            
            self.event_map[es] = es_id
            self.rev_event_map[es_id] = es
            self._event_objects[es_id] = new_event
            
            # Usar o NOVO objeto Event na transição, não string
            renamed_trs.append((q, new_event, q2))

        # 3) Constrói o DFA id-específico com objetos Event corretos
        self.supervisor = dfa(renamed_trs, initial_state(self._sup_gen), f"sup_id_{self.id}")
        self._trs_id = list(transitions(self.supervisor))
        self._state = initial_state(self.supervisor)

        # 4) (Opcional) ROS — desligado por padrão
        if self.enable_ros:
            import rospy
            from std_msgs.msg import String
            self.ros=rospy.init_node(self.name, anonymous=False)
            self.pub_state  = rospy.Publisher(f"/{self.name}/state", String, queue_size=10, latch=True)
            self.pub_events = rospy.Publisher(f"/{self.name}/possible_events", String, queue_size=10, latch=True)
            self.pub_marked = rospy.Publisher(f"/{self.name}/is_marked", String, queue_size=10, latch=True)
            self.pub_enabled_events = rospy.Publisher(f"/{self.name}/enabled_events", String, queue_size=10, latch=True)
            # NEW: assina o barramento de eventos
            self.sub_event = rospy.Subscriber("/event", String, self._on_event, queue_size=50)
            rospy.sleep(0.3)
            self._publish_ros()


    # método novo na classe:
    def _on_event(self, msg):
        """Recebe eventos do Control Panel via /event (String)."""
        ev = str(msg.data or "")
        if ev == "ping":
            # re-publica o snapshot para o painel
            self._publish_ros()
            return
        # aplica se for do meu id e estiver habilitado no estado
        _ = self.step(ev)  # step já filtra por id e verifica a transição
        
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
        """
        if not self._should_process(ev):
            return False
        
        # Converter string para objeto Event correspondente
        event_obj = self._event_objects.get(ev)
        if event_obj is None:
            return False
            
        s = str(self._state)
        for (q, e, d) in self._trs_id:
            if str(q) == s and e == event_obj:
                self._state = d
                if self.enable_ros:
                    self._publish_ros()
                return True
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

    # Extra útil em integrações físicas
    def to_generic(self, ev_with_id: str) -> str:
        return self.rev_event_map.get(ev_with_id, ev_with_id)

    def run(self):
        """Loop principal (apenas quando ROS está habilitado)."""
        if not self.enable_ros:
            return
        import rospy
        rospy.spin()


















































        

