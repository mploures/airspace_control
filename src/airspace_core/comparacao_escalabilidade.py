import os
import sys
import time
import re
from typing import List, Dict, Any, Union, Set, Tuple
import networkx as nx
# --- Caminho p/ achar graph/ ao executar via ROS ou direto ---
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)


from graph.gerar_grafo import carregar_grafo_txt 
# --- IMPORTES SOLICITADOS ---
from airspace_core.UTM import UTMModel
from airspace_core.controlador_vant import GenericVANTModel , VANTInstance
from ultrades.automata import *

def calcular_metricas_escalabilidade_custom(grafo_path: str, num_vants: int = 2, init_node: str = "VERTIPORT_0"):
    """
    Experimento comparativo entre abordagem escalável vs monolítica
    """
    print("\n" + "="*80)
    print("EXPERIMENTO COMPARATIVO: ESCALÁVEL vs MONOLÍTICO")
    print("="*80)
    
    # =====================================================================
    # CENÁRIO 1: ABORDAGEM ESCALÁVEL
    # =====================================================================
    print("\n🔧 CONFIGURANDO CENÁRIO 1 (Abordagem Escalável)...")
    
    # 1. Criar modelo genérico dos VANTs (uma vez só)
    inicio_vant_gen = time.time()
    vant_model_gen = GenericVANTModel(grafo_path, init_node)
    tempo_vant_gen = time.time() - inicio_vant_gen
    
    print(f"   ✅ GenericVANTModel criado em {tempo_vant_gen:.2f}s")
    print(f"   📊 Estados do supervisor genérico: {len(states(vant_model_gen.supervisor_mono))}")
    
    # 2. Criar UTM central
    inicio_utm = time.time()
    utm_model = UTMModel(grafo_path, init_node, num_agent=num_vants)
    tempo_utm = time.time() - inicio_utm
    
    print(f"   ✅ UTMModel criado em {tempo_utm:.2f}s")
    print(f"   📊 Estados do supervisor UTM: {len(states(utm_model.supervisor_mono))}")
    
    # 3. Criar instâncias dos VANTs (apenas renomeação, sem nova síntese)
    inicio_instancias = time.time()
    vant_instances = []
    for i in range(1, num_vants + 1):
        vant_instance = VANTInstance(
            model=vant_model_gen,
            id_num=i,
            supervisor_mono=vant_model_gen.supervisor_mono,
            enable_ros=False
        )
        vant_instances.append(vant_instance)
    tempo_instancias = time.time() - inicio_instancias
    
    print(f"   ✅ {num_vants} VANTInstances criados em {tempo_instancias:.2f}s")
    
    # Métricas do Cenário 1
    tempo_total_cenario1 = tempo_vant_gen + tempo_utm + tempo_instancias
    estados_totais_cenario1 = len(states(vant_model_gen.supervisor_mono)) + len(states(utm_model.supervisor_mono))
    
    print(f"\n📈 CENÁRIO 1 (Escalável) - RESULTADOS:")
    print(f"   ⏱️  Tempo total: {tempo_total_cenario1:.2f}s")
    print(f"   🏗️  Estados totais: {estados_totais_cenario1}")
    print(f"   📝 Sínteses realizadas: 2 (1 GenericVANTModel + 1 UTMModel)")
    
    # =====================================================================
    # CENÁRIO 2: ABORDAGEM MONOLÍTICA
    # =====================================================================
    print("\n🔧 CONFIGURANDO CENÁRIO 2 (Abordagem Monolítica)...")
    
    inicio_mono = time.time()
    
    # Criar um modelo monolítico que combina tudo
    class ModeloMonolitico:
        def __init__(self, grafo_txt: str, init_node: str, num_vants: int):
            self.grafo_txt = grafo_txt
            self.init_node = init_node
            self.num_vants = num_vants
            self.plantas_totais = []
            self.specs_totais = []
            
            # Carregar grafo
            G_in, _ = carregar_grafo_txt(grafo_txt)
            self.G = self._to_multidigraph_dirigido(G_in)
            
            # Gerar alfabeto completo com sufixos
            self.eventos = self._gerar_alfabeto_completo()
            
            # Construir todos os autômatos
            self._construir_automatos_monoliticos()
            
            # Calcular supervisor monolítico
            self.supervisor_mono = monolithic_supervisor(self.plantas_totais, self.specs_totais)
        
        @staticmethod
        def _to_multidigraph_dirigido(G_undirected):
            H = nx.MultiDiGraph()
            H.add_nodes_from(G_undirected.nodes(data=True))
            for u, v, d in G_undirected.edges(data=True):
                H.add_edge(u, v, key=0, **(d or {}))
                H.add_edge(v, u, key=0, **(d or {}))
            return H
        
        def _gerar_alfabeto_completo(self):
            """Gerar alfabeto com eventos para todos os VANTs"""
            eventos = {}
            
            # Eventos do UTM (sem sufixo)
            for u, v, k, data in self.G.edges(keys=True, data=True):
                for nome in (f"pega_{u}{v}", f"pega_{v}{u}"):
                    if nome not in eventos:
                        eventos[nome] = event(nome, controllable=True)
            for n in self.G.nodes():
                nb = f"bloqueia_{n}"; nd = f"desbloqueia_{n}"
                if nb not in eventos: eventos[nb] = event(nb, controllable=True)
                if nd not in eventos: eventos[nd] = event(nd, controllable=True)
            
            # Eventos dos VANTs (com sufixos)
            for vant_id in range(1, self.num_vants + 1):
                # Eventos de movimento
                for u, v, k, data in self.G.edges(keys=True, data=True):
                    for nome_base in [f"pega_{u}{v}", f"pega_{v}{u}", f"libera_{u}{v}", f"libera_{v}{u}"]:
                        nome_com_sufixo = f"{nome_base}_{vant_id}"
                        ctrl = not nome_base.startswith("libera_")
                        eventos[nome_com_sufixo] = event(nome_com_sufixo, controllable=ctrl)
                
                # Eventos de trabalho
                for n in self.G.nodes():
                    tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                    if tipo in {"FORNECEDOR", "CLIENTE"}:
                        for nome_base in [f"comeca_trabalho_{n}", f"fim_trabalho_{n}"]:
                            nome_com_sufixo = f"{nome_base}_{vant_id}"
                            ctrl = nome_base.startswith("comeca_")
                            eventos[nome_com_sufixo] = event(nome_com_sufixo, controllable=ctrl)
                
                # Eventos de carregamento
                for n in self.G.nodes():
                    tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                    if tipo in {"ESTACAO"}:
                        for nome_base in [f"carregar_{n}", f"fim_carregar_{n}"]:
                            nome_com_sufixo = f"{nome_base}_{vant_id}"
                            ctrl = nome_base.startswith("carregar_")
                            eventos[nome_com_sufixo] = event(nome_com_sufixo, controllable=ctrl)
                
                # Eventos globais
                for nome_base in ["aceita_tarefa", "rejeita_tarefa", "termina_tarefa", 
                                "check_vivacidade", "bateria_baixa"]:
                    nome_com_sufixo = f"{nome_base}_{vant_id}"
                    ctrl = nome_base in ["aceita_tarefa", "rejeita_tarefa", "termina_tarefa", "check_vivacidade"]
                    eventos[nome_com_sufixo] = event(nome_com_sufixo, controllable=ctrl)
            
            return eventos
        
        def _tipo_norm(self, x):
            return str(x).strip().upper()
        
        def ev(self, nome):
            return self.eventos[nome]
        
        def _construir_automatos_monoliticos(self):
            """Construir todas as plantas e especificações para sistema monolítico"""
            
            # ========== PLANTAS/SPECS DO UTM ==========
            self._construir_utm_automatos()
            
            # ========== PLANTAS/SPECS DOS VANTS ==========
            for vant_id in range(1, self.num_vants + 1):
                self._construir_vant_automatos(vant_id)
        
        def _construir_utm_automatos(self):
            """Autômatos do UTM (iguais ao UTMModel)"""
            # Mapa do UTM
            state_vertices = {}
            for n in self.G.nodes():
                s = state(str(n), marked=(n == self.init_node))
                state_vertices[n] = s
            
            initial = state_vertices.get(self.init_node, next(iter(state_vertices.values())))
            trs = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                su = state_vertices[u]; sv = state_vertices[v]
                trs.append((su, self.ev(f"pega_{u}{v}"), sv))
                trs.append((sv, self.ev(f"pega_{v}{u}"), su))
            A_mapa = dfa(trs, initial, "Mapa_UTM")
            self.plantas_totais.append(A_mapa)
            
            # Planta bloqueio admin
            bloqueio = state("Bloqueio_Global", marked=True)
            trs = []
            for n in self.G.nodes():
                e1 = self.ev(f"bloqueia_{n}"); e2 = self.ev(f"desbloqueia_{n}")
                if e1 is not None: trs.append((bloqueio, e1, bloqueio))
                if e2 is not None: trs.append((bloqueio, e2, bloqueio))
            A_bloq = accessible(dfa(trs, bloqueio, "Planta_Bloqueio"))
            self.plantas_totais.append(A_bloq)
            
            # Specs de controle de bloqueio
            for v in self.G.nodes():
                nao_bloqueado = state(f"vert_{v}_nao_bloqueado", marked=True)
                bloqueado = state(f"vert_{v}_bloqueado")
                trs = []
                e_block = self.ev(f"bloqueia_{v}"); e_unblock = self.ev(f"desbloqueia_{v}")
                
                if e_block is not None: trs.append((nao_bloqueado, e_block, bloqueado))
                if e_unblock is not None: trs.append((bloqueado, e_unblock, nao_bloqueado))
                
                # Eventos de movimento
                for u in self.G.predecessors(v):
                    e_in = self.ev(f"pega_{u}{v}")
                    if e_in is not None: trs.append((nao_bloqueado, e_in, nao_bloqueado))
                
                A = accessible(dfa(trs, nao_bloqueado, f"spec_vert_{v}_controle_bloqueio"))
                self.specs_totais.append(A)
            
            # Automato bloqueio
            Desbloqueado = state("Desbloqueado", marked=True)
            Bloqueado = state("Bloqueado")
            trs = []
            for n in self.G.nodes():
                e1 = self.ev(f"bloqueia_{n}"); e2 = self.ev(f"desbloqueia_{n}")
                if e1 is not None: trs.append((Desbloqueado, e1, Bloqueado))
                if e2 is not None: trs.append((Bloqueado, e2, Desbloqueado))
            A_bloq_spec = dfa(trs, Desbloqueado, "Bloqueio")
            self.specs_totais.append(A_bloq_spec)
            
            # Specs mutex de vértice
            Sigma_total = set(self.eventos.values())
            for v, data in self.G.nodes(data=True):
                if "VERTIPORT" in str(v).upper():
                    continue
                    
                livre = state(f"vert_{v}_livre", marked=True)
                ocupado = state(f"vert_{v}_ocupado")
                trs = []
                
                eventos_ocupacao = set()
                eventos_liberacao = set()
                
                for u in set(self.G.predecessors(v)):
                    e_in = self.ev(f"pega_{u}{v}")
                    if e_in is not None:
                        trs.append((livre, e_in, ocupado))
                        eventos_ocupacao.add(e_in)
                
                for w in set(self.G.successors(v)):
                    e_out = self.ev(f"pega_{v}{w}")
                    if e_out is not None:
                        trs.append((ocupado, e_out, livre))
                        eventos_liberacao.add(e_out)
                
                # Auto-transições
                for e in Sigma_total:
                    if e not in eventos_ocupacao:
                        trs.append((livre, e, livre))
                
                for e in Sigma_total:
                    if e not in eventos_ocupacao and e not in eventos_liberacao:
                        trs.append((ocupado, e, ocupado))
                
                A = accessible(dfa(trs, livre, f"S_vert_{v}_mutex_ocupacao"))
                self.specs_totais.append(A)
        
        def _construir_vant_automatos(self, vant_id: int):
            """Autômatos para um VANT específico"""
            sufixo = f"_{vant_id}"
            
            # ========== PLANTAS DO VANT ==========
            
            # 1. Movimento
            Parado = state(f"Parado{sufixo}", marked=True)
            Movendo = state(f"Movendo{sufixo}")
            trs = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                pega_uv = self.ev(f"pega_{u}{v}{sufixo}")
                pega_vu = self.ev(f"pega_{v}{u}{sufixo}")
                libera_uv = self.ev(f"libera_{u}{v}{sufixo}")
                libera_vu = self.ev(f"libera_{v}{u}{sufixo}")
                
                trs.extend([
                    (Parado, pega_uv, Movendo), (Movendo, libera_uv, Parado),
                    (Parado, pega_vu, Movendo), (Movendo, libera_vu, Parado),
                ])
            A_mov = dfa(trs, Parado, f"movimento{sufixo}")
            self.specs_totais.append(A_mov)
            
            # 2. Arestas (plantas)
            for u, v, k, data in self.G.edges(keys=True, data=True):
                pega_uv = self.ev(f"pega_{u}{v}{sufixo}")
                libera_uv = self.ev(f"libera_{u}{v}{sufixo}")
                
                livre_1 = state(f"livre_{u}{v}{sufixo}", marked=True)
                ocupado_1 = state(f"ocupado_{u}{v}{sufixo}")
                A1 = dfa([(livre_1, pega_uv, ocupado_1), (ocupado_1, libera_uv, livre_1)], 
                         livre_1, f"aresta_{u}{v}{sufixo}")
                self.plantas_totais.append(A1)
                
                livre_2 = state(f"livre_{v}{u}{sufixo}", marked=True)
                ocupado_2 = state(f"ocupado_{v}{u}{sufixo}")
                A2 = dfa([(livre_2, self.ev(f"pega_{v}{u}{sufixo}"), ocupado_2), 
                         (ocupado_2, self.ev(f"libera_{v}{u}{sufixo}"), livre_2)], 
                         livre_2, f"aresta_{v}{u}{sufixo}")
                self.plantas_totais.append(A2)
            
            # 3. Modos
            geral = state(f"geral{sufixo}", marked=True)
            trs = []
            for n in self.G.nodes():
                tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                if tipo in {"FORNECEDOR", "CLIENTE"}:
                    s_trab = state(f"trabalhando_{n}{sufixo}")
                    e_ini = self.ev(f"comeca_trabalho_{n}{sufixo}")
                    e_fim = self.ev(f"fim_trabalho_{n}{sufixo}")
                    trs.append((geral, e_ini, s_trab))
                    trs.append((s_trab, e_fim, geral))
                if tipo in {"ESTACAO"}:
                    s_c = state(f"carregando_{n}{sufixo}")
                    e_ini = self.ev(f"carregar_{n}{sufixo}")
                    e_fim = self.ev(f"fim_carregar_{n}{sufixo}")
                    trs.append((geral, e_ini, s_c))
                    trs.append((s_c, e_fim, geral))
            A_modos = dfa(trs, geral, f"modos{sufixo}")
            self.plantas_totais.append(A_modos)
            
            # 4. Modelos suporte
            s_com = state(f"com_ok{sufixo}", marked=True)
            Acom = dfa([
                (s_com, self.ev(f"aceita_tarefa{sufixo}"), s_com),
                (s_com, self.ev(f"rejeita_tarefa{sufixo}"), s_com),
                (s_com, self.ev(f"termina_tarefa{sufixo}"), s_com),
            ], s_com, f"comunicacao{sufixo}")
            
            s_vivo = state(f"vivo{sufixo}", marked=True)
            Aviv = dfa([(s_vivo, self.ev(f"check_vivacidade{sufixo}"), s_vivo)], 
                      s_vivo, f"vivacidade{sufixo}")
            
            s_bat = state(f"bat{sufixo}", marked=True)
            Abat = dfa([(s_bat, self.ev(f"bateria_baixa{sufixo}"), s_bat)], 
                      s_bat, f"bateria{sufixo}")
            self.plantas_totais.append(Abat)
            
            # 5. Mapa do VANT
            initial_vant = None
            state_vertices_vant = {}
            for n in self.G.nodes():
                s = state(f"{n}{sufixo}", marked=(n == self.init_node))
                state_vertices_vant[n] = s
                if n == self.init_node: initial_vant = s
            
            if initial_vant is None:
                first = next(iter(self.G.nodes()))
                initial_vant = state_vertices_vant[first]
            
            trs = []
            for u, v, k, data in self.G.edges(keys=True, data=True):
                su = state_vertices_vant[u]
                sv = state_vertices_vant[v]
                trs.append((su, self.ev(f"pega_{u}{v}{sufixo}"), sv))
                trs.append((sv, self.ev(f"pega_{v}{u}{sufixo}"), su))
            A_mapa_vant = dfa(trs, initial_vant, f"Mapa{sufixo}")
            self.specs_totais.append(A_mapa_vant)
            
            # ========== ESPECIFICAÇÕES DO VANT ==========
            
            # 6. Bateria movimento
            s_norm = state(f"bat_normal{sufixo}", marked=True)
            s_low = state(f"bat_baixa{sufixo}")
            e_low = self.ev(f"bateria_baixa{sufixo}")
            trs = [(s_norm, e_low, s_low), (s_low, e_low, s_low)]
            for n in self.G.nodes():
                tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                if tipo in {"ESTACAO"}:
                    e_ini = self.ev(f"carregar_{n}{sufixo}")
                    trs.extend([(s_low, e_ini, s_norm), (s_norm, e_ini, s_norm)])
            A_bat_mov = dfa(trs, s_norm, f"MovimentoBateria{sufixo}")
            self.specs_totais.append(accessible(A_bat_mov))
            
            # 7. Localização tarefas
            for n in self.G.nodes():
                tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                if tipo not in {"FORNECEDOR", "CLIENTE", "ESTACAO"}: continue
                s_in = state(f"dentro_{n}{sufixo}")
                s_out = state(f"fora_{n}{sufixo}", marked=True)
                trs = []
                for x in self.G.neighbors(n):
                    trs.append((s_out, self.ev(f"libera_{x}{n}{sufixo}"), s_in))
                for x in self.G.neighbors(n):
                    trs.append((s_in, self.ev(f"pega_{n}{x}{sufixo}"), s_out))
                if tipo in {"FORNECEDOR", "CLIENTE"}:
                    trs.append((s_in, self.ev(f"comeca_trabalho_{n}{sufixo}"), s_in))
                if tipo == "ESTACAO":
                    trs.append((s_in, self.ev(f"carregar_{n}{sufixo}"), s_in))
                A_loc = dfa(trs, s_out, f"loc_{n}{sufixo}")
                self.specs_totais.append(A_loc)
            
            # 8. Workflow
            s_pick = state(f"pick{sufixo}", marked=False)
            s_place = state(f"place{sufixo}", marked=False)
            s_base = state(f"vantport{sufixo}", marked=True)
            trs = []
            for n in self.G.nodes():
                tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                if tipo in {"FORNECEDOR"}: 
                    trs.append((s_base, self.ev(f"comeca_trabalho_{n}{sufixo}"), s_pick))
                if tipo in {"CLIENTE"}: 
                    trs.append((s_pick, self.ev(f"comeca_trabalho_{n}{sufixo}"), s_place))
                if tipo in {"VERTIPORT"}:
                    for u, v, k, data in self.G.in_edges(n, keys=True, data=True):
                        trs.append((s_place, self.ev(f"pega_{u}{n}{sufixo}"), s_base))
                        trs.append((s_base, self.ev(f"pega_{u}{n}{sufixo}"), s_base))
            A_work = dfa(trs, s_base, f"work_flow{sufixo}")
            self.specs_totais.append(A_work)
            
            # 9. Fim de carga
            s_apto = state(f"apto_carregar{sufixo}", marked=True)
            s_carregou = state(f"carregou_precisa_sair{sufixo}")
            trs = []
            for n in self.G.nodes():
                tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                if tipo == "ESTACAO":
                    e_ini_carregar = self.ev(f"carregar_{n}{sufixo}")
                    for x in self.G.neighbors(n):
                        e_saida = self.ev(f"pega_{n}{x}{sufixo}")
                        if e_saida is not None:
                            trs.append((s_carregou, e_saida, s_apto))
                            trs.append((s_apto, e_saida, s_apto))
                    trs.append((s_apto, e_ini_carregar, s_carregou))
            A_fim_carga = dfa(trs, s_apto, f"fim_de_carga{sufixo}")
            self.specs_totais.append(A_fim_carga)
            
            # 10. Tarefa completa
            for n in self.G.nodes():
                tipo = self._tipo_norm(self.G.nodes[n].get("tipo", ""))
                if tipo not in {"FORNECEDOR", "CLIENTE", "ESTACAO", "VERTIPORT"}: 
                    continue
                s_pode = state(f"pode_sair_{n}{sufixo}", marked=True)
                s_trab = state(f"trabalhando_{n}{sufixo}")
                trs = []
                
                eventos_saida = [self.ev(f"pega_{n}{x}{sufixo}") for x in self.G.neighbors(n)]
                for e_saida in eventos_saida:
                    trs.append((s_pode, e_saida, s_pode))
                
                e_ini = None; e_fim = None
                if tipo in {"FORNECEDOR", "CLIENTE"}:
                    e_ini = self.ev(f"comeca_trabalho_{n}{sufixo}")
                    e_fim = self.ev(f"fim_trabalho_{n}{sufixo}")
                elif tipo in {"ESTACAO"}:
                    e_ini = self.ev(f"carregar_{n}{sufixo}")
                    e_fim = self.ev(f"fim_carregar_{n}{sufixo}")
                
                if e_ini and e_fim:
                    trs.append((s_pode, e_ini, s_trab))
                    trs.append((s_trab, e_ini, s_trab))
                    trs.append((s_trab, e_fim, s_pode))
                
                A_tarefa = dfa(trs, s_pode, f"tarefa_completa_{n}{sufixo}")
                self.specs_totais.append(A_tarefa)
    
    # Criar e calcular modelo monolítico
    modelo_mono = ModeloMonolitico(grafo_path, init_node, num_vants)
    tempo_mono = time.time() - inicio_mono
    
    # Métricas do Cenário 2
    tempo_total_cenario2 = tempo_mono
    estados_totais_cenario2 = len(states(modelo_mono.supervisor_mono))
    
    print(f"\n📈 CENÁRIO 2 (Monolítico) - RESULTADOS:")
    print(f"   ⏱️  Tempo total: {tempo_total_cenario2:.2f}s")
    print(f"   🏗️  Estados totais: {estados_totais_cenario2}")
    print(f"   📝 Sínteses realizadas: 1 (monolítica completa)")
    
    # =====================================================================
    # COMPARAÇÃO FINAL
    # =====================================================================
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO FINAL")
    print("="*80)
    
    economia_tempo = ((tempo_total_cenario2 - tempo_total_cenario1) / tempo_total_cenario2) * 100
    economia_estados = ((estados_totais_cenario2 - estados_totais_cenario1) / estados_totais_cenario2) * 100
    
    print(f"⏱️  TEMPO DE COMPUTAÇÃO:")
    print(f"   Escalável: {tempo_total_cenario1:.2f}s")
    print(f"   Monolítico: {tempo_total_cenario2:.2f}s")
    print(f"   🔥 Economia: {economia_tempo:+.1f}%")
    
    print(f"\n🏗️  TAMANHO DO AUTÔMATO (Estados):")
    print(f"   Escalável: {estados_totais_cenario1} estados")
    print(f"   Monolítico: {estados_totais_cenario2} estados")
    print(f"   🔥 Redução: {economia_estados:+.1f}%")
    
    print(f"\n📝 SÍNTESES REALIZADAS:")
    print(f"   Escalável: 2 sínteses (modelos independentes)")
    print(f"   Monolítico: 1 síntese (modelo completo)")
    
    print(f"\n🎯 CONCLUSÃO:")
    if economia_tempo > 0 and economia_estados > 0:
        print(f"   ✅ A abordagem escalável é MAIS EFICIENTE em tempo e espaço!")
    else:
        print(f"   ⚠️  A abordagem monolítica pode ser mais eficiente para sistemas pequenos")
    
    return {
        'cenario1': {'tempo': tempo_total_cenario1, 'estados': estados_totais_cenario1},
        'cenario2': {'tempo': tempo_total_cenario2, 'estados': estados_totais_cenario2},
        'economia_tempo_percent': economia_tempo,
        'economia_estados_percent': economia_estados
    }

if __name__ == "__main__":
    
    # --- CONFIGURAÇÃO ---
    _GRAFO_PATH = "/home/mploures/catkin_ws/src/airspace_control/src/airspace_core/grafo_teste_escalabilidade.txt"
    _NUM_VANTS_PARA_COMPARACAO = 3
    
    # 1. CRIAÇÃO/VERIFICAÇÃO DO ARQUIVO DE GRAFO
    _DIR = os.path.dirname(_GRAFO_PATH)
    if _DIR and not os.path.exists(_DIR):
        os.makedirs(_DIR, exist_ok=True)
        
    print(f"[INFO] Criando/Verificando arquivo de grafo em: {_GRAFO_PATH}")
    with open(_GRAFO_PATH, 'w') as f:
        f.write("tipo,label,(x,y),conexoes\n") 
        f.write("LOGICO,LOGICO_0,(100,100),VERTIPORT_0,ESTACAO_0,FORNECEDOR_0,CLIENTE_0\n")
        f.write("VERTIPORT,VERTIPORT_0,(50,50),LOGICO_0\n")
        f.write("ESTACAO,ESTACAO_0,(150,50),LOGICO_0\n")
        f.write("FORNECEDOR,FORNECEDOR_0,(50,150),LOGICO_0\n")
        f.write("CLIENTE,CLIENTE_0,(150,150),LOGICO_0\n")
    print("[INFO] Arquivo de grafo criado com sucesso.")
    
    # 2. Executa a comparação
    resultados = calcular_metricas_escalabilidade_custom(_GRAFO_PATH, num_vants=_NUM_VANTS_PARA_COMPARACAO, init_node="VERTIPORT_0")
    
    print(f"\n💾 Resultados salvos: {resultados}")