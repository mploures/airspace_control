import networkx as nx
from typing import Dict, Any, Tuple, List, Set
import os, sys, re 

# --- Caminho p/ achar graph/ ao executar via ROS ou direto ---
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)

from ultrades.automata import *
from graph.gerar_grafo import carregar_grafo_txt  

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









