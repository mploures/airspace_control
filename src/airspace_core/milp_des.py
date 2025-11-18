from collections import defaultdict, deque

from airspace_core.extract_automaton_matrices import * 
import numpy as np
from gurobipy import *
from pathlib import Path
from typing import Dict, List
from math import inf
import scipy.sparse as sp
import time
import os
import threading # Importação necessária para o Mutex

# =========================
# GESTÃO GLOBAL GUROBI E MUTEX (NOVA SEÇÃO)
# =========================

# Mutex para garantir acesso exclusivo ao Gurobi em ambientes concorrentes (multiprocessos/threads)
# Este objeto deve ser criado UMA ÚNICA VEZ na inicialização do sistema.
GLOBAL_MILP_LOCK = threading.Lock() 

# Ambiente Gurobi (Criado UMA ÚNICA VEZ). 
# A criação/liberação repetida do Env causa o erro "Model has already been freed" em concorrência.
# É uma boa prática inicializar isso antes de qualquer otimização.
try:
    GLOBAL_GUROBI_ENV = Env(empty=True)
    GLOBAL_GUROBI_ENV.setParam("OutputFlag", 0) # Mude para 1 se quiser ver o output do Gurobi
    GLOBAL_GUROBI_ENV.setParam("TimeLimit", 30.0) 
    GLOBAL_GUROBI_ENV.start()
except Exception as e:
    # Captura erro de licença ou inicialização e define como None
    print(f"ERRO CRÍTICO ao inicializar GLOBAL_GUROBI_ENV: {e}")
    GLOBAL_GUROBI_ENV = None


# =========================
# Constantes do E.O.I. (GLOBAL, conforme seu código)
# =========================
BIGM = 100.0
EPSILON_W = 0.7        # Menor peso para insetivo
BETA_INCENTIVE = 10.0  # Peso do incentivo (deve ser positivo)
ALPHA_TIME = 0.4       # Peso explícito para tempo
ALPHA_STATE = 0.3      # Peso explícito para estados


def new_sub_automato_propriedade(G, e, Nc):
    """
    Versão otimizada para Nc grande.
    [... código omitido para brevidade, sem alterações ...]
    """

    # ---------------------------------------------------------
    # 1) Pré-processa transições: índice por origem
    # ---------------------------------------------------------
    transicoes = transitions(G)

    por_origem = defaultdict(list)   # orig -> lista de transições (orig, ev, dest)
    for (orig, ev, dest) in transicoes:
        por_origem[orig].append((orig, ev, dest))

    estado_inicial = e

    # BFS: cada estado recebe sua profundidade
    fila = deque([(estado_inicial, 0)])
    profundidade = {estado_inicial: 0}

    recorte_trans = []  # transições do sub-autômato

    while fila:
        estado, d = fila.popleft()
        if d == Nc:
            continue

        for (orig, ev, dest) in por_origem.get(estado, []):
            recorte_trans.append((orig, ev, dest))
            if dest not in profundidade:
                profundidade[dest] = d + 1
                fila.append((dest, d + 1))

    # ---------------------------------------------------------
    # 2) Correção iterativa da propriedade algébrica
    # ---------------------------------------------------------
    def aplica_correcao(trans_list, depth, max_iter=20):
        # [... código omitido para brevidade, sem alterações ...]
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

                if not Adj_i or not Dest_ev:
                    continue

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
    """Calcula estados alcançáveis para o horizonte H, ignorando inviáveis."""
    n_ = A_csr.shape[0]
    banned = np.zeros(n_, dtype=bool)
    if inviaveis is not None and inviaveis.size:
        banned[inviaveis] = True
    reach_ = []
    cur = np.array([start], dtype=np.int32)
    if banned[start]:
        cur = np.array([], dtype=np.int32)
    reach_.append(cur)
    
    A_csr = sp.csr_matrix(A_csr, dtype=np.float32)

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


def otimizador(Sup, estado_inicial_recorte, janela, cost_dictionary, list_eventos_interesse, list_eventos_proibidos):
    """
    1. Calcula o sub-autômato (recorte) e preenche a matriz de custos W.
    2. Constrói e resolve o MILP MPSC, AGORA APENAS OTIMIZANDO CUSTO DE ESTADO E INCENTIVO.
    
    ---
    MUDANÇA: Usa GLOBAL_MILP_LOCK e GLOBAL_GUROBI_ENV.
    ---
    """
    
    if GLOBAL_GUROBI_ENV is None:
        print("[ERRO] Otimizador não inicializado. GLOBAL_GUROBI_ENV é None.")
        return [], -1
        
    H = janela
    print(f"[LOG-MILP] 1. Iniciando otimizador para Horizonte H={H}")

    # 1. Recortar o Autômato (Mantido)
    recorte = new_sub_automato_propriedade(Sup, estado_inicial_recorte, H)
    print("[LOG-MILP] 1.1. Sub-autômato (recorte) criado com sucesso.")
    
    # 2. Extrair Matrizes (Mantido)
    k = 3  
    resultado_matrices = extract_automaton_matrices(recorte, k) 
    A_np, B_np, C_np, W, D_np, event_dict, state_index = resultado_matrices
    print(f"[LOG-MILP] 1.2. Matrizes extraídas. Dimensões: N={A_np.shape[0]}, M={C_np.shape[1]}.")

    # 3. Formatar Matrizes (Mantido)
    A_csr = sp.csr_matrix(A_np, dtype=np.float32)
    B_csr = sp.csr_matrix(B_np, dtype=np.float32)
    C_csr = sp.csr_matrix(C_np, dtype=np.float32)
    D_csr = sp.csr_matrix(D_np, dtype=np.float32)
    
    n = A_csr.shape[0]
    m = C_csr.shape[1] 
    event_names = list(event_dict.keys())
    
    # 4. Preencher matriz W corretamente (Mantido)
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
    print("[LOG-MILP] 1.3. Matriz de Custos W (E, Tf, D) preenchida.")

    # 5. Vetor de custos ponderado
    pesos_E_D_somados = ALPHA_TIME + ALPHA_STATE 
    pesos_E_D = np.array([ALPHA_TIME/pesos_E_D_somados, ALPHA_STATE/pesos_E_D_somados]) 
    
    W_ED = W[:, [0, 2]] 
    w_bar = (W_ED @ pesos_E_D).astype(np.float32) 
    print("[LOG-MILP] 1.4. Vetor de custo de estado ponderado w_bar (apenas E e D) calculado.")
    
    # 6. Índices dos Eventos (Mantido)
    name_to_idx = {nm: idx for idx, nm in enumerate(event_names)}
    I_indices = np.array([name_to_idx[nm] for nm in list_eventos_interesse if nm in name_to_idx], dtype=np.int32)
    m_I = len(I_indices)
    P_indices = np.array([name_to_idx[nm] for nm in list_eventos_proibidos if nm in name_to_idx], dtype=np.int32) 
    m_P = len(P_indices)

    # 7. Pré-cálculos (Mantido)
    inviaveis_cols = np.where((C_csr.indptr[1:] - C_csr.indptr[:-1]) == 0)[0].astype(np.int32) 
    reach = compute_reach(A_csr, H, start=0, inviaveis=inviaveis_cols)
    pos = [{int(j): k for k, j in enumerate(reach[t])} for t in range(H+1)]
    print(f"[LOG-MILP] 1.5. Estados Alcançáveis (reach) calculados. Total de estados no horizonte: {sum(len(r) for r in reach)}.")
    
    # =========================
    # MODELO GUROBI - BLOCO MUTEX
    # =========================
    event_seq = []
    model_status = GRB.LOADED # Inicializa com um status padrão

    # OBTEM LOCK para acessar Gurobi
    with GLOBAL_MILP_LOCK:
        print("[LOG-MILP] 2. Configurando Modelo Gurobi (MILP).")
        model = None # Garante que model é local

        try:
            # Criação do modelo usa o ambiente global
            model = Model("mpsc_eoi_sem_tempo", env=GLOBAL_GUROBI_ENV)
            # O TimeLimit e OutputFlag já estão configurados no GLOBAL_GUROBI_ENV

            # Variáveis 
            x = [model.addMVar(len(reach[t]), vtype=GRB.BINARY, name=f"x_{t}") for t in range(H+1)]
            u = model.addMVar((H, m), vtype=GRB.BINARY, name="u")
            
            if m_I > 0:
                tau = model.addMVar((H, m_I), vtype=GRB.BINARY, name="tau")
            else:
                tau = None
            print(f"[LOG-MILP] 2.1. Variáveis x ({H+1} vetores), u ({H}x{m}) e tau ({'presente' if m_I > 0 else 'ausente'}) adicionadas.")

            # =========================
            # RESTRIÇÕES 
            # =========================
            print("[LOG-MILP] 3. Adicionando Restrições.")
            
            # Restrições I. Estado e Transição
            if 0 in pos[0]:
                model.addConstr(x[0][pos[0][0]] == 1.0, name="init_x")
            
            for t in range(H):
                model.addConstr(x[t].sum() == 1.0, name=f"state_onehot_t{t}")
                model.addConstr(u[t, :].sum() == 1.0, name=f"event_onehot_t{t}")

            if H < A_csr.shape[0]: 
                model.addConstr(x[H].sum() == 1.0, name=f"state_onehot_t{H}")

            for t in range(H):
                rt = reach[t]
                if rt.size > 0:
                    for j in range(m):
                        coeffs = np.array([C_csr[state_global, j] for state_global in rt])
                        if np.sum(coeffs) == 0:
                            model.addConstr(u[t, j] == 0.0, name=f"event_disabled_t{t}_e{j}")
                        else:
                            model.addConstr(u[t, j] <= x[t] @ coeffs, name=f"event_feas_t{t}_e{j}")

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
            
            model.update() # Garante que todas as restrições são registradas
            print(f"[LOG-MILP] 3.1. Total de {model.numConstrs} restrições adicionadas.")

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
            print("[LOG-MILP] 4. Função Objetivo configurada (Minimizar Custo de Estado + Incentivo).")


            # =========================
            # SOLUÇÃO
            # =========================
            print("[LOG-MILP] 5. Otimização iniciada...")
            model.optimize()
            model_status = model.status # Salva o status

            if model_status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
                print(f"[LOG-MILP] 5.1. Solução encontrada. Status: {model_status}.")
                
                u_sol = u.X
                seq_idx = [np.argmax(u_sol[t, :]) for t in range(H)] 
                event_seq = [event_names[i] for i in seq_idx]
                
                cost_states_val = ALPHA_STATE * cost_states_E_D.getValue()
                
                cost_incentive_val = cost_incentive.getValue() if m_I > 0 and tau is not None else cost_incentive
                
                print(f"[✓] Solução encontrada (H={H}):")
                print(f"    Objetivo: {model.objVal:.2f}")
                print(f"    Custo Estados (E, D): {cost_states_val:.2f}")
                print(f"    Incentivo: {cost_incentive_val:.2f}")
                print(f"    Sequência de Eventos: {event_seq}")

            else:
                print(f"[LOG-MILP] 5.1. Otimização falhou. Status: {model_status}.")
                print(f"[×] Otimização falhou. Status: {model_status}")

        except Exception as e:
            # Captura exceções durante a otimização (incluindo "Model has already been freed" se persistir)
            print(f"[ERRO NO GUROBI] {e}")
            model_status = -1

        finally:
            # Libera SOMENTE o modelo. O ambiente (env) é liberado apenas no desligamento global.
            if model is not None:
                try:
                    model.dispose()
                    print("[LOG-MILP] 6. Recursos do Gurobi liberados.")
                except Exception as e:
                    # Este erro é o que você estava vendo. Agora ele deve ser menos frequente.
                    print(f"[LOG-MILP] 6. Erro ao liberar recursos: {e}")
                    pass
        
    return event_seq, model_status
