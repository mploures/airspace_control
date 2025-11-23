from collections import defaultdict, deque
from airspace_core.extract_automaton_matrices import * 
import numpy as np
from gurobipy import *
from pathlib import Path
from typing import Dict, List, Optional
from math import inf
import scipy.sparse as sp
import time
import os
import threading 

# =========================
# GESTÃO GLOBAL GUROBI E MUTEX 
# =========================

GLOBAL_MILP_LOCK = threading.Lock() 

# Armazenamento Global da Solução Anterior para Warm Start
GLOBAL_LAST_U_SEQUENCE: Optional[np.ndarray] = None 
GLOBAL_LAST_EVENT_NAMES: Optional[List[str]] = None

# Ambiente Gurobi (Criado UMA ÚNICA VEZ). 
try:
    GLOBAL_GUROBI_ENV = Env(empty=True)
    GLOBAL_GUROBI_ENV.setParam("OutputFlag", 0) 
    # AJUSTE 1: Redução do TimeLimit para resposta rápida (e.g., 5 segundos)
    GLOBAL_GUROBI_ENV.setParam("TimeLimit", 5.0) 
    # AJUSTE 2: Aumento do MIPGap para encontrar uma solução "boa o suficiente" mais rapidamente
    GLOBAL_GUROBI_ENV.setParam("MIPGap", 0.001) # 0.1% de tolerância
    GLOBAL_GUROBI_ENV.start()
except Exception as e:
    print(f"ERRO CRÍTICO ao inicializar GLOBAL_GUROBI_ENV: {e}")
    GLOBAL_GUROBI_ENV = None


# =========================
# Constantes do E.O.I. (GLOBAL)
# =========================
BIGM = 100.0
EPSILON_W = 0.7       
BETA_INCENTIVE = 10.0 
ALPHA_TIME = 0.4      
ALPHA_STATE = 0.3     


# As funções 'new_sub_automato_propriedade' e 'compute_reach'
# permanecem inalteradas, pois já são otimizadas.
# ----------------------------------------------------------------------------------

def new_sub_automato_propriedade(G, e, Nc):
    """
    Versão otimizada para Nc grande. (Não alterada)
    """
    # ---------------------------------------------------------
    # 1) Pré-processa transições, BFS e Recorte (Otimizado)
    # ---------------------------------------------------------
    transicoes = transitions(G)

    por_origem = defaultdict(list)
    for (orig, ev, dest) in transicoes:
        por_origem[orig].append((orig, ev, dest))

    estado_inicial = e

    fila = deque([(estado_inicial, 0)])
    profundidade = {estado_inicial: 0}

    recorte_trans = []  # transições do sub-autômato

    while fila:
        estado, d = fila.popleft()
        if d == Nc:
            continue

        for trans in por_origem.get(estado, []):
            orig, ev, dest = trans
            recorte_trans.append(trans)
            if dest not in profundidade:
                profundidade[dest] = d + 1
                fila.append((dest, d + 1))

    # ---------------------------------------------------------
    # 2) Correção iterativa da propriedade algébrica
    # ---------------------------------------------------------
    def aplica_correcao(trans_list, depth, max_iter=20):
        for _ in range(max_iter):
            adj = defaultdict(set)             
            dest_por_evento = defaultdict(set)
            delta = {}                         
            por_evento_dest = defaultdict(list)

            for (orig, ev, dest) in trans_list:
                adj[orig].add(dest)
                dest_por_evento[ev].add(dest)
                por_evento_dest[(ev, dest)].append(orig)
                if (orig, ev) not in delta:
                    delta[(orig, ev)] = dest

            to_remove = set()

            for (orig, ev), dest_can in delta.items():
                Adj_i = adj.get(orig, set())
                Dest_ev = dest_por_evento.get(ev, set())

                S = Adj_i & Dest_ev

                if len(S) <= 1:
                    continue

                extras = S - {dest_can}

                for dest_extra in extras:
                    for o2 in por_evento_dest.get((ev, dest_extra), []):
                        if depth.get(o2, 0) > depth.get(orig, 0):
                            to_remove.add((o2, ev, dest_extra))

            if not to_remove:
                break

            trans_list = [t for t in trans_list if t not in to_remove]

        return trans_list

    trans_corrigidas = aplica_correcao(recorte_trans, profundidade)

    # ---------------------------------------------------------
    # 3) Estados mortos e ε-loops
    # ---------------------------------------------------------
    epsolon = event("epslon", controllable=False) 

    origs = {t[0] for t in trans_corrigidas}
    dests = {t[2] for t in trans_corrigidas}
    estados_alcancaveis = set(profundidade.keys()) 
    estados_com_saida = origs
    estados_mortos = estados_alcancaveis - estados_com_saida

    for st in estados_mortos:
        trans_corrigidas.append((st, epsolon, st))

    # ---------------------------------------------------------
    # 4) Criar o novo autômato corrigido
    # ---------------------------------------------------------
    new_automaton = dfa(
        trans_corrigidas,
        estado_inicial,
        f"Sub_{str(estado_inicial)}_{str(G)}"
    )

    return new_automaton

def compute_reach(A_csr, H, start=0, inviaveis=None):
    """Calcula estados alcançáveis para o horizonte H, ignorando inviáveis. (Não alterada)"""
    n_ = A_csr.shape[0]
    banned = np.zeros(n_, dtype=bool)
    if inviaveis is not None and inviaveis.size:
        banned[inviaveis] = True
    reach_ = []
    cur = np.array([start], dtype=np.int32)
    if banned[start]:
        cur = np.array([], dtype=np.int32)
    reach_.append(cur)
    
    for _ in range(H):
        if cur.size == 0:
            nxt = np.array([], dtype=np.int32)
        else:
            A_sub = A_csr[cur, :]
            nxt = A_sub.indices
            
            nxt = np.unique(nxt)

            if nxt.size:
                nxt = nxt[~banned[nxt]].astype(np.int32)
        
        reach_.append(nxt)
        cur = nxt
    return reach_


# ----------------------------------------------------------------------------------

def otimizador(Sup, estado_inicial_recorte, janela, cost_dictionary, list_eventos_interesse, list_eventos_proibidos):
    """
    Versão Otimizada com Warm Start e Mutex Seguro.
    """
    global GLOBAL_LAST_U_SEQUENCE, GLOBAL_LAST_EVENT_NAMES
    
    if GLOBAL_GUROBI_ENV is None:
        print("[ERRO] Otimizador não inicializado. GLOBAL_GUROBI_ENV é None.")
        return [], -1
        
    H = janela
    print(f"[LOG-MILP] 1. Iniciando otimizador para Horizonte H={H}")

    # =================================================================
    # ETAPAS FORA DO MUTEX: PRÉ-PROCESSAMENTO INTENSIVO DE CPU/MEMÓRIA
    # =================================================================

    # 1. Recortar e Extrair Matrizes
    start_recorte = time.time()
    recorte = new_sub_automato_propriedade(Sup, estado_inicial_recorte, H)
    k = 3  
    resultado_matrices = extract_automaton_matrices(recorte, k) 

    A_csr, B_csr, C_csr, W, D_np, event_dict, state_index = resultado_matrices
    print(f"[LOG-MILP] 1.1. Recorte e Extração concluídos em: {time.time() - start_recorte:.4f}s.")
    
    n = A_csr.shape[0]
    m = C_csr.shape[1] 
    event_names = list(event_dict.keys())
    
    # 3. Preencher matriz W e Vetor de custo ponderado (w_bar)
    Q_recorte = list(states(recorte)) 
    for estado in Q_recorte:
        estado_str = str(estado)
        if estado_str in cost_dictionary:
            custo_E, custo_Tf, custo_D = cost_dictionary[estado_str]
        else:
            custo_E, custo_Tf, custo_D = (0.0, 0.0, 0.0)
        
        i = state_index[estado]
        W[i, 0] = custo_E   
        W[i, 1] = custo_Tf  
        W[i, 2] = custo_D   

    pesos_E_D_somados = ALPHA_TIME + ALPHA_STATE 
    pesos_E_D = np.array([ALPHA_TIME/pesos_E_D_somados, ALPHA_STATE/pesos_E_D_somados]) 
    
    W_ED = W[:, [0, 2]] 
    w_bar = (W_ED @ pesos_E_D).astype(np.float32) 
    
    # 4. Índices dos Eventos
    name_to_idx = {nm: idx for idx, nm in enumerate(event_names)}
    I_indices = np.array([name_to_idx[nm] for nm in list_eventos_interesse if nm in name_to_idx], dtype=np.int32)
    m_I = len(I_indices)
    P_indices = np.array([name_to_idx[nm] for nm in list_eventos_proibidos if nm in name_to_idx], dtype=np.int32) 
    m_P = len(P_indices)

    # 5. Pré-cálculos de alcançabilidade
    start_reach = time.time()
    inviaveis_cols = np.where((C_csr.indptr[1:] - C_csr.indptr[:-1]) == 0)[0].astype(np.int32) 
    reach = compute_reach(A_csr, H, start=0, inviaveis=inviaveis_cols)
    pos = [{int(j): k for k, j in enumerate(reach[t])} for t in range(H+1)]
    print(f"[LOG-MILP] 1.5. Pré-cálculo de alcançabilidade concluído em: {time.time() - start_reach:.4f}s.")
    
    # =================================================================
    # ETAPAS DENTRO DO MUTEX: INTERAÇÃO COM O GUROBI (MODELAGEM/SOLUÇÃO)
    # =================================================================
    event_seq = []
    model_status = GRB.LOADED 

    with GLOBAL_MILP_LOCK:
        print("[LOG-MILP] 2. Configurando Modelo Gurobi (MILP) DENTRO DO MUTEX.")
        start_model = time.time()
        model = None 

        try:
            model = Model("mpsc_eoi_sem_tempo", env=GLOBAL_GUROBI_ENV)

            # Variáveis
            x = [model.addMVar(len(reach[t]), vtype=GRB.BINARY, name=f"x_{t}") for t in range(H+1)]
            u = model.addMVar((H, m), vtype=GRB.BINARY, name="u")
            
            if m_I > 0:
                tau = model.addMVar((H, m_I), vtype=GRB.BINARY, name="tau")
            else:
                tau = None

            # =========================
            # RESTRIÇÕES - CORRIGIDAS
            # =========================
            
            # I. Estado e Transição
            if 0 in pos[0]:
                model.addConstr(x[0][pos[0][0]] == 1.0, name="init_x")
            
            # Restrições One-Hot
            for t in range(H):
                model.addConstr(x[t].sum() == 1.0, name=f"state_onehot_t{t}")
                model.addConstr(u[t, :].sum() == 1.0, name=f"event_onehot_t{t}")

            if H < A_csr.shape[0]: 
                model.addConstr(x[H].sum() == 1.0, name=f"state_onehot_t{H}")

            # Restrições de Disponibilidade (versão original)
            for t in range(H):
                rt = reach[t]
                if rt.size > 0:
                    for j in range(m):
                        coeffs = np.array([C_csr[state_global, j] for state_global in rt])
                        if np.sum(coeffs) == 0:
                            model.addConstr(u[t, j] == 0.0, name=f"event_disabled_t{t}_e{j}")
                        else:
                            model.addConstr(u[t, j] <= x[t] @ coeffs, name=f"event_feas_t{t}_e{j}")

            # Restrições de Dinâmica (versão original)
            for t in range(H):
                rt = reach[t]
                rtp1 = reach[t+1]
                if rtp1.size == 0: continue
                for idx_next, state_next in enumerate(rtp1):
                    sources = []
                    for idx_curr, state_curr in enumerate(rt):
                        for event_idx in range(m):
                            if (A_csr[state_curr, state_next] > 0 and 
                                B_csr[event_idx, state_next] > 0 and
                                C_csr[state_curr, event_idx] > 0):
                                sources.append((idx_curr, event_idx))
                    if sources:
                        lhs = x[t+1][idx_next]
                        rhs = quicksum(x[t][idx_curr] * u[t][event_idx] for idx_curr, event_idx in sources)
                        model.addConstr(lhs == rhs, name=f"dyn_t{t}_s{state_next}")
                    else:
                        model.addConstr(x[t+1][idx_next] == 0.0, name=f"dyn_unreachable_t{t}_s{state_next}")

            # Restrições III. Eventos Proibidos
            if m_P > 0:
                for p_idx in P_indices:
                    model.addConstr(u[:, p_idx].sum() == 0.0, name=f"event_prohibited_e{p_idx}")

            # Restrições IV. E.O.I. 
            if m_I > 0 and tau is not None:
                for i_idx in range(m_I):
                    e_idx = I_indices[i_idx]
                    for t in range(H):
                        if t == 0:
                            u_past = 0.0
                        else:
                            u_past = quicksum(u[tt, e_idx] for tt in range(t))
                        model.addConstr(tau[t, i_idx] <= u[t, e_idx], name=f"tau_upper1_e{e_idx}_t{t}")
                        model.addConstr(tau[t, i_idx] <= 1 - u_past, name=f"tau_upper2_e{e_idx}_t{t}")
                        model.addConstr(tau[t, i_idx] >= u[t, e_idx] - u_past, name=f"tau_lower_e{e_idx}_t{t}")
                    model.addConstr(tau[:, i_idx].sum() <= 1.0, name=f"tau_unique_e{e_idx}")
            
            model.update() 
            print(f"[LOG-MILP] 3. Construção do modelo Gurobi concluída em: {time.time() - start_model:.4f}s.")
            print(f"[LOG-MILP] 3.1. Total de {model.numConstrs} restrições adicionadas.")

            # =========================
            # WARM START (APÓS MODELAGEM)
            # =========================
            if GLOBAL_LAST_U_SEQUENCE is not None and GLOBAL_LAST_EVENT_NAMES is not None:
                u_prev = GLOBAL_LAST_U_SEQUENCE
                event_names_prev = GLOBAL_LAST_EVENT_NAMES
                H_prev = u_prev.shape[0]

                # Tenta mapear eventos do modelo anterior para o atual
                prev_to_curr_map = {name: name_to_idx.get(name) for name in event_names_prev}

                # Aplica Warm Start (Assumindo que a próxima janela desliza de 1 passo)
                # O evento ótimo no tempo t do modelo anterior é a dica para o tempo t-1 no modelo atual
                for t in range(H):
                    if t + 1 < H_prev:
                        event_idx_prev = np.argmax(u_prev[t + 1, :])
                        event_name_prev = event_names_prev[event_idx_prev]
                        
                        curr_idx = prev_to_curr_map.get(event_name_prev)
                        
                        if curr_idx is not None and curr_idx < m:
                            # Define o evento ótimo do passo t+1 anterior como dica para o passo t atual
                            u[t, curr_idx].Start = 1.0
                            print(f"[LOG-MILP] Warm Start: u[{t},{curr_idx}] = 1.0 (Evento: {event_name_prev})")

            # =========================
            # OBJETIVO
            # =========================
            
            cost_states_E_D = quicksum(
                x[t][idx] * w_bar[state_global] 
                for t in range(H) 
                for idx, state_global in enumerate(reach[t])
            )
            
            if m_I > 0 and tau is not None:
                cost_incentive = quicksum(
                    -1 * (H - t) * tau[t, i_idx] * BETA_INCENTIVE 
                    for i_idx in range(m_I) 
                    for t in range(H)
                )
            else:
                cost_incentive = 0.0 

            final_objective = ALPHA_STATE * cost_states_E_D + cost_incentive
            model.setObjective(final_objective, GRB.MINIMIZE)
            print("[LOG-MILP] 4. Função Objetivo configurada.")


            # =========================
            # SOLUÇÃO
            # =========================
            print("[LOG-MILP] 5. Otimização iniciada...")
            start_optimize = time.time()
            model.optimize()
            model_status = model.status 
            
            print(f"[LOG-MILP] 5. Otimização finalizada em: {time.time() - start_optimize:.4f}s.")

            # =========================
            # PÓS-PROCESSAMENTO (DENTRO DO MUTEX PARA ATUALIZAR GLOBAL)
            # =========================
            if model_status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
                print(f"[LOG-MILP] 5.1. Solução encontrada. Status: {model_status}.")
                
                # Armazena a nova solução em GLOBAL_LAST_U_SEQUENCE
                u_sol = u.X
                GLOBAL_LAST_U_SEQUENCE = u_sol
                GLOBAL_LAST_EVENT_NAMES = event_names
                
                seq_idx = [np.argmax(u_sol[t, :]) for t in range(H)] 
                event_seq = [event_names[i] for i in seq_idx]
                
                cost_states_val = ALPHA_STATE * cost_states_E_D.getValue()
                cost_incentive_val = cost_incentive.getValue() if m_I > 0 and tau is not None else 0.0
                
                print(f"[✓] Solução encontrada (H={H}):")
                print(f"    Objetivo: {model.objVal:.2f}")
                print(f"    Custo Estados (E, D): {cost_states_val:.2f}")
                print(f"    Incentivo: {cost_incentive_val:.2f}")
                print(f"    Sequência de Eventos: {event_seq}")

            else:
                print(f"[LOG-MILP] 5.1. Otimização falhou. Status: {model_status}.")
                print(f"[×] Otimização falhou. Status: {model_status}")
                # Não atualiza a variável global se falhar

        except Exception as e:
            print(f"[ERRO NO GUROBI] {e}")
            model_status = -1

        finally:
            if model is not None:
                try:
                    # O dispose deve estar DENTRO do Mutex para ser seguro,
                    # conforme sua proposta.
                    model.dispose()
                    print("[LOG-MILP] 6. Recursos do Gurobi liberados.")
                except Exception as e:
                    print(f"[LOG-MILP] 6. Erro ao liberar recursos: {e}")
                    pass
        
    return event_seq, model_status