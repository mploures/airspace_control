
import networkx as nx
import matplotlib.pyplot as plt
import re

# -----------------------------------------------------------------------------
# Função para ler grafo.txt
# -----------------------------------------------------------------------------


def carregar_grafo_txt(caminho_arquivo):
    G = nx.Graph()
    pos = {}

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        linhas = [l.strip() for l in f.readlines() if l.strip()]

    # Ignora o cabeçalho
    for linha in linhas[1:]:
        # Extrai tipo, label, posição e conexões com regex
        # Exemplo de linha:
        # LOGICO,LOGICO_0,(1934,1074),LOGICO_3,LOGICO_1,ESTACAO_0
        match = re.match(r'^([^,]+),([^,]+),\(([^,]+),([^)]+)\),(.*)$', linha)
        if not match:
            print(f"[AVISO] Linha ignorada (formato incorreto): {linha}")
            continue

        tipo_no = match.group(1).strip()
        label = match.group(2).strip()
        x = float(match.group(3).strip())
        y = float(match.group(4).strip())
        pos[label] = (x, y)

        G.add_node(label, tipo=tipo_no)

        # Processa conexões (se existirem)
        resto = match.group(5).strip()
        if resto:
            conectados = [p.strip() for p in resto.split(',') if p.strip()]
            for c in conectados:
                if c != label:
                    G.add_edge(label, c)

    return G, pos


# -----------------------------------------------------------------------------
# Função para desenhar o grafo
# -----------------------------------------------------------------------------
def desenhar_grafo(G, pos, titulo="Grafo Logístico (a partir do grafo.txt)"):
    plt.figure(figsize=(12, 8))

    # Classificação dos nós por tipo
    logicos   = [n for n, d in G.nodes(data=True) if d["tipo"] == "LOGICO"]
    clientes  = [n for n, d in G.nodes(data=True) if d["tipo"] == "CLIENTE"]
    fornecedores = [n for n, d in G.nodes(data=True) if d["tipo"] == "FORNECEDOR"]
    estacoes  = [n for n, d in G.nodes(data=True) if d["tipo"] == "ESTACAO"]
    vantports = [n for n, d in G.nodes(data=True) if d["tipo"] == "VANTPORT"]

    # Desenho das arestas
    nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.6)

    # Desenho dos nós (cada tipo com cor/forma diferente)
    nx.draw_networkx_nodes(G, pos, nodelist=logicos, node_color="#8a8a8a", node_size=40, label="Lógicos")
    nx.draw_networkx_nodes(G, pos, nodelist=estacoes, node_color="#1f77b4", node_shape="s", node_size=160, label="Estações")
    nx.draw_networkx_nodes(G, pos, nodelist=fornecedores, node_color="#2ca02c", node_shape="o", node_size=160, label="Fornecedores")
    nx.draw_networkx_nodes(G, pos, nodelist=clientes, node_color="#ff7f0e", node_shape="D", node_size=160, label="Clientes")
    nx.draw_networkx_nodes(G, pos, nodelist=vantports, node_color="#d62728", node_shape="^", node_size=180, label="Vantports")

    plt.legend()
    plt.title(titulo)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Execução direta
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    caminho = "graph/sistema_logistico/grafo.txt"  # coloque o caminho do seu arquivo aqui
    G, pos = carregar_grafo_txt(caminho)
    print(f"Grafo carregado: {len(G.nodes())} nós, {len(G.edges())} arestas")
    desenhar_grafo(G, pos)