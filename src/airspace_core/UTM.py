import networkx as nx
import os, sys, re 
import threading
from typing import Dict, Any, Tuple, List, Set, Union
from collections import deque
import rospy
from std_msgs.msg import String

# --- Caminho p/ achar graph/ ao executar via ROS ou direto ---
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)

from ultrades.automata import *
from graph.gerar_grafo import carregar_grafo_txt  

_RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")

# --- Funções Auxiliares para Cálculo do Supervisor ---

def pre_mapear_eventos_factíveis(automato: Any) -> Dict[State, Set[Event]]:
    """
    Pré-calcula e retorna um dicionário mapeando cada objeto State do autômato 
    ao conjunto de Eventos factíveis a partir desse estado.
    
    ASSUME que transitions(automato) retorna uma LISTA DE TUPLAS (origem, evento, destino).
    """
    eventos_por_estado: Dict[State, Set[Event]] = {s: set() for s in states(automato)}
    
    # 1. Obtém a lista de transições (origem, evento, destino)
    lista_transicoes = transitions(automato)

    # 2. Itera sobre todas as transições e popula o dicionário
    for origem, evento, destino in lista_transicoes:
        # Se a origem for um estado válido (e não None), adiciona o evento ao conjunto
        if origem in eventos_por_estado:
            eventos_por_estado[origem].add(evento)
            
    return eventos_por_estado

def pre_calcular_eventos_proibidos(supervisor: Any, plantas: List) -> Dict[State, Set[Event]]:
    """
    Calcula o conjunto de eventos proibidos (E_Plantas \ E_Supervisor) para cada estado
    acessível do Supervisor, usando o pré-mapeamento de eventos.
    """
    eventos_proibidos_por_estado = {}
    
    # Pré-mapeamento do Supervisor: {State_Supervisor: {Eventos}}
    mapa_eventos_supervisor = pre_mapear_eventos_factíveis(supervisor)
    
    # Pré-mapeamento de todas as Plantas: [ {nome_estado_str: objeto_state} ]
    mapas_estado_plantas = [{str(s): s for s in states(planta)} for planta in plantas]
    
    # Pré-mapeamento de Eventos de todas as Plantas: [ {State_Planta: {Eventos}} ]
    mapas_eventos_plantas = [pre_mapear_eventos_factíveis(planta) for planta in plantas]
    
    # Itera sobre todos os estados alcançáveis do supervisor
    for estado_supervisor_atual in states(supervisor):
        
        estado_nome = str(estado_supervisor_atual)
        nomes_estados_componentes = estado_nome.split('|')
        
        eventos_possiveis_plantas = set()
        
        # 1. União dos Eventos Factíveis em todas as Plantas (E_Plantas)
        for i, nome_estado_planta in enumerate(nomes_estados_componentes):
            
            if i >= len(plantas):
                break
                
            mapa_estado = mapas_estado_plantas[i]
            mapa_eventos = mapas_eventos_plantas[i]
            
            estado_planta = mapa_estado.get(nome_estado_planta)
            
            if estado_planta is not None:
                # Usa o mapa pré-calculado para obter os eventos
                eventos_planta_i = mapa_eventos.get(estado_planta, set())
                eventos_possiveis_plantas.update(eventos_planta_i)

        # 2. Eventos Factíveis no Supervisor (E_Supervisor)
        # Usa o mapa pré-calculado
        eventos_permitidos_supervisor = mapa_eventos_supervisor.get(estado_supervisor_atual, set())
            
        # 3. Cálculo dos Eventos Proibidos (E_Proibidos = E_Plantas \ E_Supervisor)
        eventos_proibidos = eventos_possiveis_plantas.difference(eventos_permitidos_supervisor)

        eventos_proibidos_por_estado[estado_supervisor_atual] = eventos_proibidos

    return eventos_proibidos_por_estado

class UTMModel:
    """
    Modelo DES da UTM Minimalista (Monolítico Computável)
    Funcionalidade: Apenas Evitação de Colisão (Mutex de Aresta) + Mapa.
    """

    # ----------------------------- Utilitários internos -----------------------------
    @staticmethod
    def _to_multidigraph_dirigido(G_undirected: nx.Graph) -> nx.MultiDiGraph:
        """Converte grafo não-dirigido em MultiDiGraph dirigido (u,v) e (v,u)."""
        H = nx.MultiDiGraph()
        H.add_nodes_from(G_undirected.nodes(data=True))
        for u, v, d in G_undirected.edges(data=True):
            H.add_edge(u, v, key=0, **(d or {}))
            H.add_edge(v, u, key=0, **(d or {}))
        return H

    def ev(self, nome: str) -> Any:
        return self.eventos[nome]

    # ----------------------------- Construtor -----------------------------
    def __init__(self, grafo_txt: str, init_node: str, num_agent=1):
        G_in, _ = carregar_grafo_txt(grafo_txt)
        self.G: nx.MultiDiGraph = self._to_multidigraph_dirigido(G_in)
        self.init_node: str = init_node
        self.grafo_txt: str = grafo_txt
        self.num_agent=num_agent
        self.dict_aresta_eventos = {}
        self.state_vertices: Dict[Any, Any] = {}
        
        # O alfabeto agora é gerado de forma minimalista
        self.eventos = self._gerar_alfabeto_utm()
        
        self.plantas=[]
        self.specs=[]
        self.Dicionario_Automatos: Dict[str, Any] = {}
        self.supervisor_mono = None

        self._automato_mapa()
        self._automato_planta_bloqueio_admin()
        self._automatos_specs_controle_bloqueio_vertice()
        self._automato_bloqueio()
        self._specs_vertice_mutex_ocupacao()

        print(f"[UTMModel] Plantas: {len(self.plantas)}")
        print(f"[UTMModel] Specs  : {len(self.specs)}")

        
        self.supervisor_mono = self.compute_monolithic_supervisor()
        self.eventos_proibidos_estado=pre_calcular_eventos_proibidos(self.supervisor_mono, self.plantas)
        self.agent_state = [initial_state(self.supervisor_mono) for _ in range(self.num_agent)]

    # ------------------------- Geração do Alfabeto Minimalista -------------------------
    def _gerar_alfabeto_utm(self) -> Dict[str, Any]:
        G = self.G
        eventos: Dict[str, Any] = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            for nome in (f"pega_{u}{v}", f"pega_{v}{u}"):
                if nome not in eventos:
                    eventos[nome] = event(nome, controllable=True)
        for n in G.nodes():
            nb = f"bloqueia_{n}"; nd = f"desbloqueia_{n}"
            if nb not in eventos: eventos[nb] = event(nb, controllable=True)
            if nd not in eventos: eventos[nd] = event(nd, controllable=True)
        return eventos

    # ------------------------- Modelo -------------------------

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
        self.plantas.append(A)

    def _automato_planta_bloqueio_admin(self):
        bloqueio = state("Bloqueio_Global", marked=True)
        trs = []
        for n in self.G.nodes():
            e1 = self.ev(f"bloqueia_{n}"); e2 = self.ev(f"desbloqueia_{n}")
            if e1 is not None: trs.append((bloqueio, e1, bloqueio))
            if e2 is not None: trs.append((bloqueio, e2, bloqueio))

        A = accessible(dfa(trs, bloqueio, "Planta_Bloqueio"))
        self.plantas.append(A)
        self.Dicionario_Automatos["Planta_Bloqueio"] = A
    
    def _automatos_specs_controle_bloqueio_vertice(self):
        G = self.G
        for v in G.nodes():
            # Estados: Não_Bloqueado (operacional) e Bloqueado (restrito)
            nao_bloqueado = state(f"vert_{v}_nao_bloqueado", marked=True)
            bloqueado = state(f"vert_{v}_bloqueado")
            trs = []
            e_block = self.ev(f"bloqueia_{v}")
            e_unblock = self.ev(f"desbloqueia_{v}")

            # 🛑 Transições de Bloqueio/Desbloqueio (Comportamento de Controle)
            if e_block is not None:
                trs.append((nao_bloqueado, e_block, bloqueado))

            if e_unblock is not None:
                trs.append((bloqueado, e_unblock, nao_bloqueado))

            # 🚀 Transições de Movimento
            eventos_movimento = []
            for u in G.predecessors(v):
                e_in = self.ev(f"pega_{u}{v}")
                if e_in is not None: eventos_movimento.append(e_in)

            for ev in set(eventos_movimento): 
                trs.append((nao_bloqueado, ev, nao_bloqueado))

            # Cria, acessibiliza e armazena o autômato de especificação
            A = accessible(dfa(trs, nao_bloqueado, f"spec_vert_{v}_controle_bloqueio"))
            self.specs.append(A)
            self.Dicionario_Automatos[f"spec_vert_{v}_controle_bloqueio"] = A

    def _automato_bloqueio(self):
        Desbloqueado = state("Desbloqueado", marked=True)
        Bloqueado = state("Bloqueado")

        trs = []
        for n in self.G.nodes():
            e1 = self.ev(f"bloqueia_{n}"); 
            e2 = self.ev(f"desbloqueia_{n}")
            if e1 is not None: trs.append((Desbloqueado, e1, Bloqueado))
            if e2 is not None: trs.append((Bloqueado, e2, Desbloqueado))


        A = dfa(trs, Desbloqueado, "Bloqueio")
        self.Dicionario_Automatos["Bloqueio"] = A
        self.specs.append(A)

    def _specs_vertice_mutex_ocupacao(self):
        """
        Especificação Mutex de Ocupação de Vértice (v).
        Regra: Apenas um agente por vez pode estar no vértice.
        
        """
        G = self.G
        
        # Conjunto de TODOS os eventos no sistema (pega + bloqueio/desbloqueio)
        Sigma_total = set(self.eventos.values())

        for v, data in G.nodes(data=True):
            
            # === BYPASS PARA VERTIPORT ===
            # Se o nó é um VERTIPORT, não criamos a especificação Mutex
            # Assumindo que o tipo de nó pode ser inferido pelo nome (e.g., "VERTIPORT_0")
            if "VERTIPORT" in str(v).upper():
                # O agente pode estar sempre "dentro" do VERTIPORT
                continue 
                
            # 1. Estados e Transições
            livre     = state(f"vert_{v}_livre", marked=True)
            ocupado   = state(f"vert_{v}_ocupado")
            trs = []
            
            eventos_ocupacao = set()
            eventos_liberacao = set()

            # 2. Ocupação (Livre -> Ocupado)
            for u in set(G.predecessors(v)):
                e_in = self.ev(f"pega_{u}{v}")
                if e_in is not None:
                    trs.append((livre, e_in, ocupado))
                    eventos_ocupacao.add(e_in)

            # 3. Liberação (Ocupado -> Livre)
            for w in set(G.successors(v)):
                e_out = self.ev(f"pega_{v}{w}")
                if e_out is not None:
                    trs.append((ocupado, e_out, livre))
                    eventos_liberacao.add(e_out)
                    
            # 4. Cálculo do Complemento (Auto-transições)
            
            # Eventos que causam transição (e não devem ser loops no seu estado de partida)
            eventos_transicionais = (eventos_ocupacao | eventos_liberacao)
            
            # Eventos que NUNCA devem ser loops (i.e., aqueles que são proibidos no estado 'ocupado')
            # Neste caso, apenas os eventos de OCUPAÇÃO são proibidos no estado 'ocupado'.
            
            
            # 4.1. Auto-transições no estado LIVRE
            # Todos os eventos do alfabeto SÃO PERMITIDOS, exceto os de OCUPAÇÃO (que causam a transição)
            for e in Sigma_total:
                if e not in eventos_ocupacao:
                    # Se for um evento de liberação ou outro movimento/controle, permanece livre
                    trs.append((livre, e, livre))

            # 4.2. Auto-transições no estado OCUPADO
            # Todos os eventos do alfabeto SÃO PERMITIDOS, exceto os de OCUPAÇÃO (pois já está ocupado)
            # e exceto os de LIBERAÇÃO (que causam a transição)
            for e in Sigma_total:
                if e not in eventos_ocupacao and e not in eventos_liberacao:
                    # Se for outro movimento (não de liberação) ou controle, permanece ocupado
                    trs.append((ocupado, e, ocupado))

            # 5. Criação do Autômato
            A = accessible(dfa(trs, livre, f"S_vert_{v}_mutex_ocupacao"))
            self.specs.append(A)
            self.Dicionario_Automatos[f"S_vert_{v}_mutex_ocupacao"] = A

    # ------------------------- Supervisor -------------------------
    def compute_monolithic_supervisor(self, force: bool = False) -> Any:
        if self.supervisor_mono is None or force:
            self.supervisor_mono = monolithic_supervisor(self.plantas, self.specs)
        return self.supervisor_mono

class UTMROSInterface:
    def __init__(self,
                 grafo_txt: str,
                 init_node: str,
                 num_agent: int = 1,
                 node_name: str = "utm_supervisor_node"):
        
        # ---------------- 1. Inicialização do Modelo DES ----------------
        self.utm_model = UTMModel(grafo_txt, init_node, num_agent)
        self.supervisor = self.utm_model.supervisor_mono
        self.eventos_proibidos_estado = self.utm_model.eventos_proibidos_estado
        self.name = node_name
        self.num_agent = num_agent 

        # NOVO: Contador para gerar IDs de tarefa sequenciais (ex: Tarefa_1, Tarefa_2)
        self.task_counter = 0

        # Controle de tarefas
        # A fila agora guarda TUPLE: (ID_COMPLETO_TAREFA, NÓS_RAW) 
        # Ex: ('Tarefa_1:F0,C0', 'F0,C0')
        self.task_queue: deque[Tuple[str, str]] = deque() 
        self.active_task_agents: Set[int] = set()  # agentes com tarefa ativa (ID do agente)
        self.task_lock = threading.Lock()     # lock para acesso à fila/conjunto
        
        # Estado individual de cada agente no supervisor monolítico
        self.agent_states = self.utm_model.agent_state 
        self.state_lock = threading.Lock() # Para acesso seguro ao estado
        
        self._factible_events_map = self._get_factible_events_map(self.supervisor)
        self.generic_event_objects = self.utm_model.eventos
        
        # ---------------- 2. Inicialização e Canais ROS ----------------
        rospy.init_node(self.name, anonymous=False)

        # 1. Publicadores
        self.pub_state        = rospy.Publisher(f"/{self.name}/state", String, queue_size=10, latch=True) 
        self.pub_global_prohibited = rospy.Publisher("/eventos_proibidos", String, queue_size=10, latch=True)
        # Este é o canal que o VANTs escutam, com o formato ID:F,C
        self.pub_tarefas      = rospy.Publisher("/tarefas", String, queue_size=10) 
        self.pub_events = rospy.Publisher(f"/{self.name}/possible_events", String, queue_size=10, latch=True)
        self.pub_enabled_events = rospy.Publisher(f"/{self.name}/enabled_events", String, queue_size=10, latch=True)
        self.pub_marked = rospy.Publisher(f"/{self.name}/is_marked", String, queue_size=10, latch=True)

        # 2. Subscribers
        self.sub_event        = rospy.Subscriber("/event", String, self._on_event, queue_size=50)
        # Este canal recebe o formato simples F,C
        self.sub_tarefas_afazer = rospy.Subscriber("/tarefas_afazer", String, self._on_tarefa_afazer, queue_size=10)

        rospy.sleep(0.5)
        self._publish_state() # Publica o estado inicial

    # ---------------- 3. Callbacks ROS (Entrada) ----------------

    def _dispatch_next_tasks_if_possible(self):
        """
        Envia tarefas da fila /tarefas_afazer para /tarefas,
        RESPEITANDO o limite de UVs com tarefa ativa.
        PUBLICA a tarefa no formato ID_TAREFA:FORNECEDOR,CLIENTE.
        """
        with self.task_lock:
            # Disponível: Slots totais (num_agent) - Agentes ativos
            available_slots = self.num_agent - len(self.active_task_agents)
            
            while available_slots > 0 and self.task_queue:
                # item_fila é (ID_COMPLETO, NÓS_RAW)
                tarefa_completa, _nodes_raw = self.task_queue.popleft()
                
                self.pub_tarefas.publish(String(data=tarefa_completa))
                rospy.loginfo(
                    f"[{self.name}] Tarefa despachada para a frota: '{tarefa_completa}'. "
                    f"Tarefas ativas (aceitas) atuais: {len(self.active_task_agents)}/{self.num_agent}"
                )
                # Diminui o contador de slots disponíveis (embora só conte quando for aceita)
                available_slots -= 1 

    def _extract_event_info(self, ev_with_id: str) -> Tuple[str, Union[int, None]]:
        """
        Extrai o nome do evento genérico e o ID do agente.
        """
        m = _RE_SUFFIX.match(ev_with_id)
        if m:
            generic_name = m.group(1)
            agent_id = int(m.group(2))
            return generic_name, agent_id
        return ev_with_id, None


    def _on_event(self, msg):
        """
        Recebe eventos do barramento /event (com ou sem ID de agente) e:
        - Atualiza contagem de tarefas aceitas/encerradas (tarefa_aceita_*, tarefa_encerrada_*)
        - Aplica eventos genéricos no supervisor monolítico (para um agente ou globalmente).
        """
        ev_with_id = str(msg.data or "").strip()
        if not ev_with_id:
            return

        # 1) Faz o parse com a regex padrão
        generic_name, agent_id = self._extract_event_info(ev_with_id)

        # 1a) Ajuste importante:
        # Se o "generic_name" NÃO existe no alfabeto do supervisor,
        # mas o evento COMPLETO (ev_with_id) existe, então tratamos
        # como um evento global da UTM sem ID de agente.
        # Exemplo: "bloqueia_F0" foi cortado pela regex em ("bloqueia_F", 0),
        # mas o alfabeto tem "bloqueia_F0", não "bloqueia_F".
        if generic_name not in self.generic_event_objects and ev_with_id in self.generic_event_objects:
            generic_name = ev_with_id
            agent_id = None

        # --- 0. Ping (pode vir com ou sem sufixo numérico, mas nos interessa o prefixo) ---
        if generic_name == "ping":
            rospy.loginfo(f"[{self.name}] Ping recebido — republishing state and events.")
            self._publish_state()
            return

        # --- 1. Eventos de TAREFA (não pertencem ao autômato DES) ---

        # Com a regex atual, "tarefa_aceita_1" -> generic_name="tarefa_aceita", agent_id=1
        if generic_name == "tarefa_aceita":
            if agent_id is not None:
                with self.task_lock:
                    # Garante que o número de tarefas ativas nunca ultrapassa num_agent
                    if agent_id not in self.active_task_agents:
                        if len(self.active_task_agents) < self.num_agent:
                            self.active_task_agents.add(agent_id)
                            rospy.loginfo(
                                f"[{self.name}] ✅ tarefa_aceita por agente {agent_id}. "
                                f"Tarefas ativas: {len(self.active_task_agents)}/{self.num_agent}"
                            )
                        else:
                            rospy.logwarn(
                                f"[{self.name}] ⚠️ tarefa_aceita_{agent_id} excederia o limite de "
                                f"{self.num_agent} tarefas ativas. Ignorando para contagem interna."
                            )
                    else:
                        rospy.loginfo(
                            f"[{self.name}] tarefa_aceita_{agent_id} recebida, "
                            "mas agente já constava com tarefa ativa."
                        )
            return

        if generic_name == "tarefa_encerrada":
            if agent_id is not None:
                is_removed = False
                with self.task_lock:
                    if agent_id in self.active_task_agents:
                        self.active_task_agents.remove(agent_id)
                        is_removed = True
                        rospy.loginfo(
                            f"[{self.name}] ✅ tarefa_encerrada por agente {agent_id}. "
                            f"Slot liberado. Tarefas ativas: {len(self.active_task_agents)}/{self.num_agent}"
                        )
                    else:
                        rospy.loginfo(
                            f"[{self.name}] tarefa_encerrada_{agent_id} recebida, "
                            "mas agente não constava com tarefa ativa."
                        )

                # Depois de liberar slot, tenta despachar nova tarefa
                if is_removed:
                    self._dispatch_next_tasks_if_possible()
            return

        # --- 2. Lógica do Supervisor DES (eventos do autômato) ---

        ev_obj = self.generic_event_objects.get(generic_name)

        # Se não é evento do autômato, ignoramos aqui (já tratamos tarefas/ping acima)
        if ev_obj is None:
            return

        # 2a) Caso de evento GLOBAL da UTM (sem ID de agente):
        #     ex.: "bloqueia_F0", "desbloqueia_C3" gerados pelo próprio UTM.
        if agent_id is None:
            with self.state_lock:
                any_change = False
                for i, current_state in enumerate(self.agent_states):
                    next_state = self._get_next_state(current_state, ev_obj)
                    if next_state is not None and next_state != current_state:
                        rospy.loginfo(
                            f"[{self.name}] Transição GLOBAL em cópia {i}: "
                            f"{current_state} --{generic_name}--> {next_state}"
                        )
                        self.agent_states[i] = next_state
                        any_change = True

                if any_change:
                    self._publish_state()
            return

        # 2b) Caso de evento associado a um agente específico (como antes)
        if agent_id < 1 or agent_id > self.num_agent:
            return

        agent_idx = agent_id - 1

        with self.state_lock:
            current_state = self.agent_states[agent_idx]
            next_state = self._get_next_state(current_state, ev_obj)

            if next_state is not None:
                rospy.loginfo(
                    f"[{self.name}] Transição Agente {agent_id}: "
                    f"{current_state} --{generic_name}--> {next_state}"
                )
                self.agent_states[agent_idx] = next_state
                self._publish_state()



    def _on_tarefa_afazer(self, msg):
        """
        Recebe tarefas a serem feitas (formato: F,C), GERA um ID, e:
        - Enfileira no formato ID:F,C em self.task_queue
        - Tenta despachar para a frota respeitando o limite de tarefas ativas.
        """
        # Formato esperado: "F0,C0"
        nodes_raw = str(msg.data or "").strip()
        if not nodes_raw:
            return

        # Verificação básica de formato (deve conter uma vírgula)
        if "," not in nodes_raw:
            rospy.logwarn(f"[{self.name}] Tarefa recebida em /tarefas_afazer com formato inválido (sem vírgula): {nodes_raw}")
            return
            
        # Geração de ID e enfileiramento ATÔMICO
        with self.task_lock:
            # 1. Incrementa o contador de tarefas
            self.task_counter += 1
            task_id = f"Tarefa_{self.task_counter}"
            
            # 2. Cria a string no formato completo para o VANT
            tarefa_completa = f"{task_id}:{nodes_raw}"
            
            rospy.loginfo(f"[{self.name}] Nova Tarefa recebida. ID Gerado: {task_id}. Enfileirando: {tarefa_completa}")
            
            # 3. Enfileira a tupla (ID_COMPLETO, NÓS_RAW)
            self.task_queue.append((tarefa_completa, nodes_raw))
        
        # Tenta despachar a partir da fila (se houver slot livre)
        self._dispatch_next_tasks_if_possible()
        
    # ---------------- 4. Lógica do Supervisor DES ----------------
    
    def _get_factible_events_map(self, automato: Any) -> Dict[Any, Set[Any]]:
        """
        Pré-calcula os eventos factíveis de cada estado do supervisor (como objetos Event).
        """
        eventos_por_estado: Dict[Any, Set[Any]] = {s: set() for s in states(automato)}
        for origem, evento, destino in transitions(automato):
            eventos_por_estado[origem].add(evento)
        return eventos_por_estado

    def _get_next_state(self, current_state: Any, event_obj: Any) -> Any:
        """Busca o próximo estado no autômato supervisor monolítico."""
        for q, e, d in transitions(self.supervisor):
            if q == current_state and e == event_obj:
                return d
        return None

    def _get_global_prohibited_events(self) -> Set[str]:
        """
        Calcula a **união** dos nomes (str) dos eventos proibidos para todos os estados 
        dos agentes que estão sendo rastreados.
        """
        global_prohibited: Set[str] = set()
        
        with self.state_lock:
            for agent_state in self.agent_states:
                events_prohibited_for_state = self.eventos_proibidos_estado.get(agent_state, set())
                
                for ev_obj in events_prohibited_for_state:
                    global_prohibited.add(str(ev_obj))

        return global_prohibited

    def _get_enabled_events(self) -> Set[str]:
        """
        Calcula os eventos HABILITADOS (Factíveis - Proibidos).
        """
        all_factible_agent_events: Set[str] = set()
        with self.state_lock:
            for agent_state in self.agent_states:
                factible_events = self._factible_events_map.get(agent_state, set())
                for ev_obj in factible_events:
                    all_factible_agent_events.add(str(ev_obj))
        
        global_prohibited = self._get_global_prohibited_events()
        
        enabled_events = all_factible_agent_events.difference(global_prohibited)
        
        return enabled_events

    def _get_blocking_events_filtered(self, candidate_events: Set[str]) -> Set[str]:
        """
        Filtra os eventos proibidos globais para incluir apenas os 
        eventos de Bloqueio/Desbloqueio (Controle de Nó).
        """
        bloqueios = set()
        for ev_name in candidate_events:
            if ev_name.startswith(("bloqueia_", "desbloqueia_")):
                bloqueios.add(ev_name)
        
        return bloqueios


    # ---------------- 5. Publicação ROS (Saída - Inalterada) ----------------

    def _publish_state(self):
        """Publica o estado atual, a lista global de eventos proibidos e os eventos de bloqueio de nó."""
        
        # --- 1. Publica o Estado dos Agentes ---
        state_strs = [str(s) for s in self.agent_states]
        state_str = "-||-".join(state_strs)
        self.pub_state.publish(String(data=state_str))
        rospy.loginfo(f"[{self.name}] Estados dos Agentes Publicados: {state_str}")
        
        # --- 2. Calcula Eventos Proibidos Globais ---
        global_prohibited = self._get_global_prohibited_events()
        global_str = ",".join(sorted(global_prohibited))
        
        self.pub_global_prohibited.publish(String(data=global_str))
        rospy.loginfo(f"[{self.name}] Eventos PROIBIDOS Globais Publicados em /eventos_proibidos: {global_str}")

        # --- 3. Publica Eventos Habilitados (Permitidos) ---
        enabled_events = self._get_enabled_events()
        blocking_events = self._get_blocking_events_filtered(enabled_events)
        enabled_str = ",".join(sorted(blocking_events))

        self.pub_events.publish(String(data=enabled_str)) 
        self.pub_enabled_events.publish(String(data=enabled_str))
        rospy.loginfo(f"[{self.name}] Eventos Habilitados (Permitidos) Publicados: {enabled_str}")

        # --- 4. Estado Marcado ---
        self.pub_marked.publish(String(data="False"))


    def run(self):
        """Loop principal do nó."""
        rospy.spin()


