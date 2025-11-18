import networkx as nx
from typing import Dict, Any, Tuple, List
import os, sys, re 

# --- Caminho p/ achar graph/ ao executar via ROS ou direto ---
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)


from ultrades.automata import *
from graph.gerar_grafo import carregar_grafo_txt  


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
    def __init__(self, grafo_txt: str, init_node: str):
        G_in, _ = carregar_grafo_txt(grafo_txt)
        self.G: nx.MultiDiGraph = self._to_multidigraph_dirigido(G_in)
        self.init_node: str = init_node
        self.grafo_txt: str = grafo_txt
        
        self.dict_aresta_eventos: Dict[Tuple[Tuple[str, str], Any], Tuple[Any, Any, Any, Any]] = {}
        self.state_vertices: Dict[Any, Any] = {}
        
        # O alfabeto agora é gerado de forma minimalista
        self.eventos: Dict[str, Any] = self._gerar_alfabeto_utm()
        
        self.plantas=[]
        self.specs=[]
        self.Dicionario_Automatos: Dict[str, Any] = {}
        self.supervisor_mono = None
        
        self._automatos_arestas()
        self._automato_mapa()
        self._automato_movimento()
        self._automatos_vertice_bloqueio()
        self._automatos_arestas_sentido()
        #self._automatos_vertice_mutex()

        
        self.supervisor_mono = self.compute_monolithic_supervisor()

    # ------------------------- Geração do Alfabeto Minimalista -------------------------
    def _gerar_alfabeto_utm(self) -> Dict[str, Any]:
        """
        Gera o alfabeto completo do UTM: eventos de aresta (pega/libera) e
        eventos de controle de nó (bloqueia/desbloqueia).
        """
        G = self.G
        eventos: Dict[str, Any] = {}
        
        # 1. Eventos de Aresta (pega/libera) - Comportamento Físico
        # (Copie a lógica de _gerar_alfabeto_minimalista)
        for u, v, k, data in G.edges(keys=True, data=True):
            for nome in (f"pega_{u}{v}", f"pega_{v}{u}", f"libera_{u}{v}", f"libera_{v}{u}"):
                if nome not in eventos:
                    # 'pega' é controlável (ação do supervisor/VANT), 'libera' é não-controlável (fim do movimento)
                    ctrl = not nome.startswith("libera_") 
                    eventos[nome] = event(nome, controllable=ctrl)

            # Associação no dict de arestas (chave simétrica) - Importante para _automatos_arestas
            chave = (tuple(sorted((u, v))), k)
            if chave not in self.dict_aresta_eventos:
                self.dict_aresta_eventos[chave] = (
                    eventos[f"pega_{u}{v}"], 
                    eventos[f"pega_{v}{u}"], 
                    eventos[f"libera_{u}{v}"], 
                    eventos[f"libera_{v}{u}"]
                )

        # 2. Eventos de Controle de Nó (bloqueia/desbloqueia) - Ações do UTM
        for n in G.nodes():
            # Evento de Bloqueio - Controlável pelo UTM
            nome_bloqueia = f"bloqueia_{n}"
            if nome_bloqueia not in eventos:
                eventos[nome_bloqueia] = event(nome_bloqueia, controllable=True)
                
            # Evento de Desbloqueio - Controlável pelo UTM
            nome_desbloqueia = f"desbloqueia_{n}"
            if nome_desbloqueia not in eventos:
                eventos[nome_desbloqueia] = event(nome_desbloqueia, controllable=True)
                
        # Armazenar o alfabeto no objeto
        self._eventos_utm = eventos
        return eventos


    # ------------------------- Modelo -------------------------

    def _automatos_vertice_bloqueio(self):
        """
        Plantas triviais: para cada vértice v, um autômato 1-estado que
        apenas diz que existem os eventos bloqueia_v / desbloqueia_v.
        A lógica de quando pode bloquear / desbloquear fica TODA na spec.
        """
        G = self.G
        for v in G.nodes():
            s = state(f"block_planta_{v}", marked=True)
            e_block   = self.ev(f"bloqueia_{v}")
            e_unblock = self.ev(f"desbloqueia_{v}")

            trs = [
                (s, e_block,   s),
                (s, e_unblock, s),
            ]

            A = dfa(trs, s, f"block_{v}")
            self.plantas.append(A)
            self.Dicionario_Automatos[f"block_{v}"] = A

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

    def _automato_movimento(self):
        Parado = state("Parado", marked=True)
        Movendo = state("Movendo_utm")
        #Movendo2 = state("Movendo2")
        #Movendo3 = state("Movendo3")
        trs = []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            pega_uv = self.ev(f"pega_{u}{v}"); pega_vu = self.ev(f"pega_{v}{u}")
            libera_uv = self.ev(f"libera_{u}{v}"); libera_vu = self.ev(f"libera_{v}{u}")
            if chave not in self.dict_aresta_eventos:
                self.dict_aresta_eventos[chave] = (pega_uv, pega_vu, libera_uv, libera_vu)
            trs.extend([
                (Parado, pega_uv, Movendo), (Movendo, libera_uv, Parado),
                #(Movendo1, pega_uv, Movendo2), (Movendo2, libera_uv, Movendo1),
                #(Movendo2, pega_uv, Movendo3), (Movendo3, libera_uv, Movendo2),


                (Parado, pega_vu, Movendo), (Movendo, libera_vu, Parado),
                #(Movendo1, pega_vu, Movendo2), (Movendo2, libera_vu, Movendo1),
                #(Movendo2, pega_vu, Movendo3), (Movendo3, libera_vu, Movendo2),
            ])
        A = dfa(trs, Parado, "movimento")
        self.Dicionario_Automatos["movimento"] = A
        self.specs.append(A)
    
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

    def _automatos_arestas_sentido(self):
        """
        Para cada aresta {u,v}_k (não-dirigida), um recurso de capacidade 1:

          Estados:
            - edge_{u}{v}_{k}_livre   (inicial, marcado)
            - edge_{u}{v}_{k}_ocupado

          Transições:
            - livre --pega_{u}{v}--> ocupado
            - livre --pega_{v}{u}--> ocupado
            - ocupado --libera_{u}{v}--> livre
            - ocupado --libera_{v}{u}--> livre

        Isso garante:
          - No máximo um VANT na aresta {u,v} (qualquer direção).
          - Não precisamos mais de tau_* nem de estados POS/NEG.
        """
        vistos = set()
        for u, v, k, data in self.G.edges(keys=True, data=True):
            chave = (tuple(sorted((u, v))), k)
            if chave in vistos:
                continue
            vistos.add(chave)

            pega_uv, pega_vu, libera_uv, libera_vu = self.dict_aresta_eventos[chave]

            livre   = state(f"edge_{u}{v}_{k}_livre", marked=True)
            ocupado = state(f"edge_{u}{v}_{k}_ocupado")

            trs = [
                (livre,   pega_uv,   ocupado),
                (livre,   pega_vu,   ocupado),
                (ocupado, libera_uv, livre),
                (ocupado, libera_vu, livre),
            ]

            A = dfa(trs, livre, f"edge_{u}{v}_{k}")
            self.specs.append(A)
            self.Dicionario_Automatos[f"edge_{u}{v}_{k}"] = A
    
    def _automatos_vertice_mutex(self):
        """
        Para cada vértice v, uma ÚNICA especificação que junta:
          - mutex (no máximo 1 ocupante em v);
          - bloqueio/desbloqueio:

            * impede novas chegadas quando bloqueado;
            * permite bloquear estando ocupado;
            * permite saída enquanto bloqueado (o agente já “dentro” pode sair).

        Estados:
          - vert_{v}_livre     (inicial, marcado)
          - vert_{v}_ocupado
          - vert_{v}_bloqueado
        """
        G = self.G

        for v in G.nodes():
            livre     = state(f"vert_{v}_livre", marked=True)
            ocupado   = state(f"vert_{v}_ocupado")
            bloqueado = state(f"vert_{v}_bloqueado")

            trs = []

            # 1) Chegadas: pega_uv com destino v (u -> v)
            tem_chegada = False
            for u in set(G.predecessors(v)):
                nome_in = f"pega_{u}{v}"
                if nome_in in self.eventos:
                    e_in = self.ev(nome_in)
                    # só pode chegar se estiver LIVRE
                    trs.append((livre, e_in, ocupado))
                    tem_chegada = True

            # Se não há chegadas em v, não faz sentido ter mutex/bloqueio
            if not tem_chegada:
                continue

            # 2) Saídas: pega_vw que saem de v (v -> w)
            for w in set(G.successors(v)):
                nome_out = f"pega_{v}{w}"
                if nome_out in self.eventos:
                    e_out = self.ev(nome_out)
                    # saída normal: ocupado -> livre
                    trs.append((ocupado,   e_out, livre))
                    # saída enquanto bloqueado: continua bloqueado
                    trs.append((bloqueado, e_out, bloqueado))

            # 3) Bloqueio / desbloqueio
            e_block   = self.ev(f"bloqueia_{v}")
            e_unblock = self.ev(f"desbloqueia_{v}")

            # Podemos bloquear tanto em LIVRE quanto em OCUPADO
            trs.append((livre,   e_block, bloqueado))
            trs.append((ocupado, e_block, bloqueado))

            # De BLOQUEADO volta para LIVRE ao desbloquear
            trs.append((bloqueado, e_unblock, livre))

            # Em BLOQUEADO não há chegadas (não adicionamos pega_uv saindo de livre/bloqueado
            # para um estado ocupado), garantindo que ninguém novo entra enquanto v está bloqueado.

            A = dfa(trs, livre, f"spec_vert_{v}_mutex_block")
            self.specs.append(A)
            self.Dicionario_Automatos[f"spec_vert_{v}_mutex_block"] = A
    
    def compute_monolithic_supervisor(self, force: bool = False) -> Any:
        if self.supervisor_mono is None or force:
            self.supervisor_mono = monolithic_supervisor(self.plantas, self.specs)
        return self.supervisor_mono
