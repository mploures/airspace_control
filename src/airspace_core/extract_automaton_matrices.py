import numpy as np
from ultrades.automata import *
# import matplotlib.pyplot as plt  # não usado; pode remover

def extract_automaton_matrices(G, k):
    """
    G: DFA da UltraDES
    k: número de componentes de custo (int ou string convertível para int)

    Retorna:
    A ∈ {0,1}^{n x n}  (adjacency matrix)
    B ∈ {0,1}^{m x n}  (event reachability matrix: linha=evento, coluna=estado destino)
    C ∈ {0,1}^{n x m}  (event availability matrix: linha=estado origem, coluna=evento)
    W ∈ ℝ^{n x k}      (state cost matrix, zeros)
    D ∈ ℝ^{m x m}      (min temporal separation, zeros)
    event_dict: {str(event): (event_obj, one_hot_vector)}
    """
    # --- Normaliza k (corrige TypeError: 'str' object cannot be interpreted as an integer)
    try:
        k = int(k)
    except Exception as err:
        raise TypeError(f"'k' must be an integer (got {k!r})") from err

    # --- Materializa iteráveis da UltraDES (corrige o erro de len())
    Q = list(states(G))
    E = list(events(G))
    T = list(transitions(G))

    n = len(Q)
    m = len(E)

    # Índices
    state_index = {q: i for i, q in enumerate(Q)}
    event_index = {e: i for i, e in enumerate(E)}

    # Matrizes booleanas compactas
    A = np.zeros((n, n), dtype=np.bool_)
    B = np.zeros((m, n), dtype=np.bool_)  # linha=evento, coluna=estado destino
    C = np.zeros((n, m), dtype=np.bool_)  # linha=estado origem, coluna=evento

    # Preenche a partir das transições
    for (q_i, sigma, q_j) in T:
        i = state_index[q_i]
        j = state_index[q_j]
        eidx = event_index[sigma]

        A[i, j] = True
        B[eidx, j] = True
        C[i, eidx] = True

    # Custos e separações temporais
    W = np.zeros((n, k), dtype=np.float32)
    D = np.zeros((m, m), dtype=np.float32)

    # One-hots por evento
    event_dict = {}
    for e in E:
        onehot = np.zeros(m, dtype=np.bool_)
        onehot[event_index[e]] = True
        event_dict[str(e)] = (e, onehot)

    return A, B, C, W, D, event_dict


def print_automaton_data(A, B, C, W, D, event_dict):
    np.set_printoptions(
        threshold=np.inf,
        linewidth=200,
        formatter={'bool': lambda x: '1' if x else '0'}
    )

    print("───── Matriz A (adjacency: n x n) ─────")
    print(A.astype(int)); print()

    print("───── Matriz B (event reachability: m x n) ─────")
    print(B.astype(int)); print()

    print("───── Matriz C (event availability: n x m) ─────")
    print(C.astype(int)); print()

    print("───── Matriz W (state costs: n x k) ─────")
    print(W); print()

    print("───── Matriz D (event separation: m x m) ─────")
    print(D); print()

    print("───── Dicionário de Eventos ─────")
    for e_str, (e_obj, onehot) in event_dict.items():
        print(f"{e_str} →")
        print(f"  Objeto: {e_obj}")
        print(f"  One-hot: {onehot.astype(int)}")
        print()


def verifica_propriedade(A, B, C):
    """
    Verifica a propriedade de determinismo algébrico:
    Para cada (estado i, evento e) disponível (C[i,e]=1), deve existir
    exatamente 1 próximo estado alcançável por e.
    """
    n, m = C.shape  # C é n x m
    # Matmul em inteiros para contar caminhos
    D = (A.astype(np.int64) @ B.T.astype(np.int64))  # n x m, D[i,e] = nº de destinos via e
    V = (C.astype(np.int64) * D)  # hadamard -> mantém contagens só onde C=1

    problemas = []
    for i in range(n):
        for e in range(m):
            if C[i, e]:
                valor_V = V[i, e]
                if valor_V != 1:
                    problemas.append({
                        'estado': i,
                        'evento': e,
                        'valor_C': int(C[i, e]),
                        'valor_V': int(valor_V),
                        'tipo': 'MÚLTIPLOS ESTADOS' if valor_V > 1 else 'NENHUM ESTADO'
                    })

    ok = np.array_equal(V, C.astype(np.int64))
    return ok, problemas


