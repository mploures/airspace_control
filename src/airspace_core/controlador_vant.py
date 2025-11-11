#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =================================================================================================
# Imports (Python, Grafo, UltraDES)
# =================================================================================================
from typing import Dict, Tuple, List, Any, Iterable, Optional
import os, sys, re
import networkx as nx

# --- Caminho p/ achar graph/ ao executar via ROS ou direto ---
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)

from airspace_core.uav_agent import VANT
from graph.gerar_grafo import carregar_grafo_txt  
from ultrades.automata import *

# =================================================================================================
# ### NOVOS IMPORTS - Lógica do Nó ROS ###
# =================================================================================================
import rospy
from std_msgs.msg import String

# =================================================================================================
# Classe 1: Modelo genérico (sem sufixo em eventos/estados)
# =================================================================================================
class GenericVANTModel:
    """
    Modelo DES genérico (sem sufixo _{id}) para um grafo e nó inicial.
    Constrói plantas e especificações e permite calcular/salvar supervisores uma única vez.
    """
    
    # ... (CÓDIGO EXISTENTE DA GenericVANTModel PERMANECE IGUAL) ...
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
    @classmethod
    def carregar_posicoes(cls, caminho_arquivo: str) -> Dict[str, Tuple[float, float]]:
        posicoes: Dict[str, Tuple[float, float]] = {}
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
                m = cls._COORD_RE.match(posicao_raw) or cls._COORD_RE.search(linha)
                if not m:
                    raise ValueError(f"Não foi possível extrair coordenadas de: {linha}")
                x = float(m.group(1)); y = float(m.group(2))
                posicoes[label] = (x, y)
        return posicoes
    # ----------------------------- Construtor -----------------------------
    def __init__(self, grafo_txt: str, init_node: str):
        G_in, _ = carregar_grafo_txt(grafo_txt)
        self.G: nx.MultiDiGraph = self._to_multidigraph_dirigido(G_in)
        self.init_node: str = init_node
        self.grafo_txt: str = grafo_txt
        self.posicoes: Dict[str, Tuple[float, float]] = self.carregar_posicoes(grafo_txt)
        self.posicao_evento: Dict[str, Tuple[Any, Tuple[float, float]]] = {}
        self.dict_aresta_eventos: Dict[Tuple[Tuple[str, str], Any], Tuple[Any, Any, Any, Any]] = {}
        self.state_vertices: Dict[Any, Any] = {}
        self.eventos: Dict[str, Any] = self._gerar_alfabeto_generico()
        self.plantas: List[Any] = []
        self.specs: List[Any] = []
        self.Dicionario_Automatos: Dict[str, Any] = {}
        self._automato_movimento()
        self._automatos_arestas()
        self._automato_modos()
        self._modelos_suporte()
        self._automato_mapa()
        self._automato_bateria_movimento()
        self._automatos_localizacao_tarefas()
        self.supervisor_mono: Optional[Any] = None
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
            if tipo in {"ESTACAO", "VERTIPORT"}:
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
            if tipo in {"ESTACAO", "VERTIPORT"}:
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
            if tipo in {"ESTACAO", "VERTIPORT"}:
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
    # ------------------------------- Supervisor / IO -------------------------------
    def compute_monolithic_supervisor(self, force: bool = False) -> Any:
        if self.supervisor_mono is None or force:
            self.supervisor_mono = monolithic_supervisor(self.plantas, self.specs)
        return self.supervisor_mono
    def save_wmod(self, path: str):
        write_wmod(self.plantas, self.specs, path)
    def save_supervisor_xml(self, path: str, ensure_supervisor: bool = True):
        if ensure_supervisor and self.supervisor_mono is None:
            self.compute_monolithic_supervisor()
        if self.supervisor_mono is None:
            raise RuntimeError("Supervisor não disponível.")
        write_xml(self.supervisor_mono, path)

# =================================================================================================
# Classe 2: Instância por VANT — modo lógico (sem ROS) por padrão
# =================================================================================================
import re
from ultrades.automata import dfa, transitions, states, initial_state, is_marked, event

class VANTInstance:
    """
    Especializa o supervisor genérico (sem sufixo) para um VANT específico (id_num),
    renomeando eventos para '<evento>_{id}'. Em modo lógico (enable_ros=False), expõe
    uma API mínima para teste: enabled_events(), step(event), state().
    """
    _RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")

    def __init__(self,
                 model: 'GenericVANTModel',
                 id_num: int,
                 supervisor_mono=None,
                 obj_vant=None,
                 enable_ros: bool = False,
                 node_name: str = None):
        self.model = model
        self.id = int(id_num)
        self.obj_vant = obj_vant
        self.enable_ros = bool(enable_ros)
        self.name = node_name or f"supervisor_vant_{self.id}"

        # 1) Recupera (ou calcula) supervisor genérico
        if supervisor_mono is None:
            if getattr(model, "supervisor_mono", None) is None:
                supervisor_mono = model.compute_monolithic_supervisor()
            else:
                supervisor_mono = model.supervisor_mono
        self._sup_gen = supervisor_mono

        # 2) Renomeia TODAS as transições do supervisor para evento_{id}
        # MAS preservando os objetos Event originais
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
            rospy.init_node(self.name, anonymous=False)
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


















































        

