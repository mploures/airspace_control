#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random 


# ===========================
# Leitura de arquivos de pontos
# ===========================
def ler_pares_xy(arquivo):
    pts = []
    with open(arquivo, "r", encoding="utf-8") as f:
        for i, linha in enumerate(f):
            if i == 0:  # pula cabeçalho
                continue
            linha = linha.strip()
            if not linha:
                continue
            x, y = linha.split(",")
            pts.append((int(x), int(y)))
    return pts

# ===========================
# Utilidades geométricas
# ===========================
def distancia(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def orient(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

def on_segment(a, b, c):
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )

def segmentos_cruzam(p1, p2, p3, p4):
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if (o1 * o2 < 0) and (o3 * o4 < 0):  # caso geral
        return True

    # casos colineares
    if o1 == 0 and on_segment(p1, p2, p3):
        return True
    if o2 == 0 and on_segment(p1, p2, p4):
        return True
    if o3 == 0 and on_segment(p3, p4, p1):
        return True
    if o4 == 0 and on_segment(p3, p4, p2):
        return True

    return False

def dentro_ret(p, x1, y1, x2, y2):
    return (x1 < p[0] < x2) and (y1 < p[1] < y2)

# ===========================
# Malha H/V a partir dos cruzamentos
# ===========================
def construir_malha_planar(cruzamentos):
    """
    Conecta apenas vizinhos imediatos:
    - por linha (mesmo y): ordenar xs e ligar pares consecutivos
    - por coluna (mesmo x): ordenar ys e ligar pares consecutivos
    """
    nodes = set(cruzamentos)
    xs_by_y = defaultdict(list)
    ys_by_x = defaultdict(list)

    for x, y in cruzamentos:
        xs_by_y[y].append(x)
        ys_by_x[x].append(y)

    for y in xs_by_y:
        xs_by_y[y].sort()
    for x in ys_by_x:
        ys_by_x[x].sort()

    adj = defaultdict(set)

    # horizontal
    for y, xs in xs_by_y.items():
        for i in range(len(xs) - 1):
            u = (xs[i], y)
            v = (xs[i + 1], y)
            adj[u].add(v)
            adj[v].add(u)

    # vertical
    for x, ys in ys_by_x.items():
        for i in range(len(ys) - 1):
            u = (x, ys[i])
            v = (x, ys[i + 1])
            adj[u].add(v)
            adj[v].add(u)

    return nodes, adj, xs_by_y, ys_by_x

# ===========================
# Imagem integral (contagem O(1))
# ===========================
def integral_from_points(points, H, W):
    grid = np.zeros((H, W), dtype=np.uint8)
    for x, y in points:
        if 0 <= x < W and 0 <= y < H:
            grid[y, x] = 1
    ii = np.zeros((H + 1, W + 1), dtype=np.int32)
    ii[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    return ii

def rect_count(ii, x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # Adicionamos +1 a x2 e y2 para incluir as coordenadas (x2, y2)
    # Note que a imagem integral ii tem dimensões (H+1, W+1)
    return int(ii[y2 + 1, x2 + 1] - ii[y1, x2 + 1] - ii[y2 + 1, x1] + ii[y1, x1])

# ===========================
# Enumeração de retângulos viáveis
# ===========================
def listar_retangulos(
    nodes, xs_by_y, ys_by_x, H, W, dist_min_borda=20, dist_min_lado=50
):
    unique_x = sorted(ys_by_x.keys())
    unique_y = sorted(xs_by_y.keys())
    nodes_set = set(nodes)

    retangulos = []
    for yi in range(len(unique_y)):
        for yj in range(yi + 1, len(unique_y)):
            y1 = unique_y[yi]
            y2 = unique_y[yj]
            for xi in range(len(unique_x)):
                for xj in range(xi + 1, len(unique_x)):
                    x1 = unique_x[xi]
                    x2 = unique_x[xj]

                    # bordas mínimas da imagem
                    if (
                        x1 < dist_min_borda
                        or x2 > W - dist_min_borda
                        or y1 < dist_min_borda
                        or y2 > H - dist_min_borda
                    ):
                        continue

                    # lados mínimos
                    if (x2 - x1) < 2 * dist_min_lado or (y2 - y1) < 2 * dist_min_lado:
                        continue

                    c1, c2, c3, c4 = (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                    # Verifica se os quatro cantos SÃO NÓS LOGICOS. 
                    # Isso garante que a malha H/V "alcance" os limites do retângulo.
                    if (
                        c1 in nodes_set
                        and c2 in nodes_set
                        and c3 in nodes_set
                        and c4 in nodes_set
                    ):
                        retangulos.append((x1, y1, x2, y2))

    return list(set(retangulos))


# ===========================
# Atribuição de tipos 
# ===========================
def atribuir_tipos(construcoes_roi, N_V, N_E, N_F, N_C, dist_min=0):
    pts = construcoes_roi[:]
    selecionados = []

    def valido(p):
        for _, q in selecionados:
            if distancia(p, q) < dist_min:
                return False
        return True

    def pick():
        if not pts:
            return None
        
        # ----------------------------------------------------
        # ALTERAÇÃO: Seleção ALEATÓRIA em vez de Farthest-First
        # ----------------------------------------------------
        
        # 1. Filtra os pontos restantes que SÃO VÁLIDOS (respeitam dist_min)
        pts_validos = [p for p in pts if valido(p)]
        
        if not pts_validos:
            # Não há pontos restantes que respeitem a distância mínima
            return None
        
        # 2. Escolhe um ponto aleatório entre os válidos
        p = random.choice(pts_validos)
        
        # 3. Remove o ponto selecionado da lista principal de pontos
        pts.remove(p)
        return p
        
        # ----------------------------------------------------


    saida = []
    for tipo, N in [
        ("VERTIPORT", N_V),
        ("ESTACAO", N_E),
        ("FORNECEDOR", N_F),
        ("CLIENTE", N_C),
    ]:
        for _ in range(N):
            # A chamada a pick() agora já garante que o ponto é válido quanto a dist_min
            p = pick() 
            if p is not None:
                # O teste 'valido(p)' dentro do loop principal é tecnicamente redundante 
                # se 'pick' já filtra, mas mantemos o mínimo de alteração na estrutura
                # do loop 'for' para não causar problemas.
                selecionados.append((tipo, p))
                saida.append((tipo, p))
            # Se pick() retorna None, significa que não há mais pontos disponíveis
            # ou nenhum ponto restante respeita a dist_min. O loop continua para o próximo tipo.
            
    return saida

# ===========================================================
# CONEXÃO DIAGONAL: especial -> vértices da célula que o contém
# ===========================================================
def _norm_edge(u, v):
    return tuple(sorted((u, v)))

def _bracketing(sorted_vals, val):
    """Retorna (low, high) vizinhos adjacentes que cercam 'val' (ou (None,None))."""
    i = bisect.bisect_left(sorted_vals, val)
    if i == 0 or i == len(sorted_vals):
        return None, None
    low = sorted_vals[i - 1]
    high = sorted_vals[i]
    if not (low < val < high):
        return None, None
    return low, high

def conectar_especial_por_diagonais(p, grid_xs, grid_ys, nodes_set, edges_set):
    """
    Encontra a célula [xL,xR]x[yT,yB] que contém p e retorna conexões
    do especial p para os **quatro vértices** dessa célula (duas diagonais),
    respeitando:
    - não cruzar nenhuma aresta existente exceto no vértice de destino;
    - não tocar nenhum outro nó lógico além do destino.
    """
    x, y = p
    xL, xR = _bracketing(grid_xs, x)
    yT, yB = _bracketing(grid_ys, y)
    if xL is None or xR is None or yT is None or yB is None:
        return []

    corners = [(xL, yT), (xR, yB), (xL, yB), (xR, yT)]  # duas diagonais
    conexoes = []

    for corner in corners:
        if corner not in nodes_set:
            continue

        # 1) não cruzar arestas existentes (permitir tocar no destino)
        cruza = False
        for (u, v) in edges_set:
            if u == corner or v == corner:
                continue  # tocar no destino é permitido
            if segmentos_cruzam(p, corner, u, v):
                cruza = True
                break
        if cruza:
            continue

        # 2) não tocar outro nó lógico no meio
        toca = False
        for w in nodes_set:
            if w == corner:
                continue
            if orient(p, corner, w) == 0 and on_segment(p, corner, w):
                toca = True
                break
        if toca:
            continue

        conexoes.append((p, corner))

    return conexoes

# ===========================
# Montagem do grafo
# ===========================
def montar_grafo(nodes_roi, edges_roi, pontos_atribuidos, conexoes_especiais):
    grafo = {}
    idmap = {n: f"LOGICO_{i}" for i, n in enumerate(sorted(nodes_roi))}

    for n, nid in idmap.items():
        grafo[nid] = {"tipo": "LOGICO", "posicao": n, "conexoes": set()}

    for (u, v) in edges_roi:
        a, b = idmap[u], idmap[v]
        grafo[a]["conexoes"].add(b)
        grafo[b]["conexoes"].add(a)

    cont = defaultdict(int)
    for tipo, p in pontos_atribuidos:
        nid = f"{tipo}_{cont[tipo]}"
        cont[tipo] += 1
        grafo[nid] = {"tipo": tipo, "posicao": p, "conexoes": set()}

    # adiciona conexões especiais (podem ser múltiplas por especial)
    for (p, n) in conexoes_especiais:
        id_log = idmap.get(n) # usar .get para garantir que 'n' está no idmap (no BBox final)
        if id_log is None:
            continue 
            
        id_esp = None
        for k, v in grafo.items():
            if v["tipo"] != "LOGICO" and v["posicao"] == p:
                id_esp = k
                break
        if id_esp:
            grafo[id_log]["conexoes"].add(id_esp)
            grafo[id_esp]["conexoes"].add(id_log)

    for k in grafo:
        grafo[k]["conexoes"] = list(grafo[k]["conexoes"])

    return grafo

# ===========================
# Desenho
# ===========================
def desenhar_grafo_mapa(imagem_path, ret, grafo, output_path):
    raio=20
    img = None
    if imagem_path and os.path.exists(imagem_path):
        img = cv2.imread(imagem_path)

    if img is None:
        x1, y1, x2, y2 = ret
        W = (x2 - x1) + 100
        H = (y2 - y1) + 100
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        def adj(p):
            return (p[0] - x1 + 50, p[1] - y1 + 50)

        draw_offset = True
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def adj(p):
            return p

        draw_offset = False

    cores = {
        "VERTIPORT": (255, 0, 0),
        "ESTACAO": (0, 0, 255),
        "FORNECEDOR": (0, 255, 0),
        "CLIENTE": (255, 255, 0),
        "LOGICO": (128, 0, 128),
    }

    # retângulo
    x1, y1, x2, y2 = ret
    if not draw_offset:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # arestas
    seen = set()
    for u, info in grafo.items():
        for v in info["conexoes"]:
            e = tuple(sorted([u, v]))
            if e in seen:
                continue
            seen.add(e)
            p1 = adj(grafo[u]["posicao"])
            p2 = adj(grafo[v]["posicao"])
            cv2.line(img, p1, p2, (0, 0, 0), 20) 

    # nós
    for nid, info in grafo.items():
        x, y = adj(info["posicao"])
        cor = cores[info["tipo"]]
        cv2.circle(img, (int(x), int(y)), raio, cor, -1)


    if output_path:
        out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, out_bgr)

    return img

def salvar_grafo_txt(grafo, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("tipo_do_no,label,posicao,labels_dos_nos_conectados\n")
        for node_id, info in grafo.items():
            x, y = info["posicao"]
            f.write(f"{info['tipo']},{node_id},({x},{y}),{','.join(info['conexoes'])}\n")
# ===========================
# Função principal
# ===========================
def criar_sistema_logistico(
    N_VERTIPORT,
    N_ESTACAO,
    N_FORNECEDOR,
    N_CLIENTE,
    arquivo_cruzamentos="pontos_cruzamentos.txt",
    arquivo_construcoes="pontos_construcoes.txt",
    imagem_mapa="mapafinal.png",
    output_dir="output",
    dist_minima_pontos=100,
    dist_minima_borda=20,
):
    os.makedirs(output_dir, exist_ok=True)

    cruzamentos = ler_pares_xy(arquivo_cruzamentos)
    construcoes = ler_pares_xy(arquivo_construcoes)

    if not cruzamentos:
        print("ERRO: sem cruzamentos.")
        return None

    # Dimensões
    if imagem_mapa and os.path.exists(imagem_mapa):
        H, W = cv2.imread(imagem_mapa).shape[:2]
    else:
        xs = [x for x, _ in cruzamentos + construcoes]
        ys = [y for _, y in cruzamentos + construcoes]
        # Adiciona margem para W e H
        W = max(xs) + dist_minima_borda * 2 + 1 
        H = max(ys) + dist_minima_borda * 2 + 1

    # 1) malha planar H/V
    nodes, adj, xs_by_y, ys_by_x = construir_malha_planar(cruzamentos)

    # 2) retângulos viáveis e escolha do melhor (menor área que atende a demanda)
    rects = listar_retangulos(
        nodes,
        xs_by_y,
        ys_by_x,
        H,
        W,
        dist_min_borda=dist_minima_borda,
        dist_min_lado=dist_minima_pontos,
    )
    if not rects:
        print("ERRO: nenhum retângulo viável encontrado.")
        return None

    ii = integral_from_points(construcoes, H, W)
    demanda = N_VERTIPORT + N_ESTACAO + N_FORNECEDOR + N_CLIENTE # Demanda é o total de pontos a serem alocados (não o dobro)

    melhor_rect_inicial, area_best = None, float('inf')
    for (x1, y1, x2, y2) in rects:
        cnt = rect_count(ii, x1, y1, x2, y2)
        if cnt >= demanda:
            area = (x2 - x1) * (y2 - y1)
            if area < area_best:
                area_best = area
                melhor_rect_inicial = (x1, y1, x2, y2)

    if melhor_rect_inicial is None:
        print("ERRO: nenhum retângulo atende à capacidade mínima de construções.")
        return None

    roi_inicial = melhor_rect_inicial
    x1, y1, x2, y2 = roi_inicial
    # print(f"ROI inicial (mínima área c/ cantos na malha): {roi_inicial} área={area_best}")


    # 3) subgrafo lógico no ROI INICIAL (e seleção das construções)
    nodes_roi_inicial = set([n for n in nodes if (x1 <= n[0] <= x2 and y1 <= n[1] <= y2)])
    
    # 4) escolhe construções na ROI inicial e atribui tipos (usando a capacidade mínima do retângulo)
    construcoes_roi_inicial = [p for p in construcoes if (x1 <= p[0] <= x2 and y1 <= p[1] <= y2)]
    pts_atribuidos = atribuir_tipos(
        construcoes_roi_inicial,
        N_VERTIPORT,
        N_ESTACAO,
        N_FORNECEDOR,
        N_CLIENTE,
        dist_min=dist_minima_pontos,
    )
    
    
    # 3.1) ALTERAÇÃO PRINCIPAL: REDEFINIR ROI para o Bounding Box (BBox) mínimo
    # Contendo todos os NÓS LÓGICOS E PONTOS ESPECIAIS contidos no ROI inicial.
    
    # NÓS LÓGICOS E PONTOS ESPECIAIS NA ROI FINAL
    todos_pontos_necessarios = list(nodes_roi_inicial) + [p for _, p in pts_atribuidos]

    if not todos_pontos_necessarios:
        print("ERRO: BBox mínima não pode ser definida (sem nós lógicos ou especiais).")
        return None
        
    xs_necessarios = [p[0] for p in todos_pontos_necessarios]
    ys_necessarios = [p[1] for p in todos_pontos_necessarios]

    # O novo ROI (BBox) deve ser limitado pelos *cruzamentos* mais próximos para não ter "espaço em branco"
    # A coordenada X do BBox final deve ser uma coordenada X de cruzamento
    # A coordenada Y do BBox final deve ser uma coordenada Y de cruzamento
    grid_xs = sorted(ys_by_x.keys())
    grid_ys = sorted(xs_by_y.keys())
    
    # BBox: Xmin, Ymin, Xmax, Ymax
    # Encontra o cruzamento X mais próximo e menor ou igual ao min(xs_necessarios)
    # Encontra o cruzamento X mais próximo e maior ou igual ao max(xs_necessarios)
    # E o mesmo para Y.
    
    # bisect_left acha o ponto de inserção. O elemento à esquerda é o floor, à direita é o ceil.
    
    # X1 (Mínimo): O maior x de cruzamento <= min(xs_necessarios)
    i = bisect.bisect_right(grid_xs, min(xs_necessarios))
    # Se i=0, significa que min(xs_necessarios) é menor que o menor cruzamento (problema de ROI inicial?)
    # Se i > 0, grid_xs[i-1] é o maior X de cruzamento <= min(xs)
    x_min_idx = i - 1 if i > 0 else 0
    
    # X2 (Máximo): O menor x de cruzamento >= max(xs_necessarios)
    i = bisect.bisect_left(grid_xs, max(xs_necessarios))
    # Se i=len(grid_xs), significa que max(xs_necessarios) é maior que o maior cruzamento.
    # Se i < len(grid_xs), grid_xs[i] é o menor X de cruzamento >= max(xs)
    x_max_idx = i if i < len(grid_xs) else len(grid_xs) - 1

    # Y1 (Mínimo): O maior y de cruzamento <= min(ys_necessarios)
    i = bisect.bisect_right(grid_ys, min(ys_necessarios))
    y_min_idx = i - 1 if i > 0 else 0
    
    # Y2 (Máximo): O menor y de cruzamento >= max(ys_necessarios)
    i = bisect.bisect_left(grid_ys, max(ys_necessarios))
    y_max_idx = i if i < len(grid_ys) else len(grid_ys) - 1
    
    # Nova ROI ajustada
    x1_final = grid_xs[x_min_idx]
    x2_final = grid_xs[x_max_idx]
    y1_final = grid_ys[y_min_idx]
    y2_final = grid_ys[y_max_idx]
    
    # Garantir que min < max (pode acontecer se houver apenas um nó lógico em X ou Y)
    if x1_final > x2_final: x1_final = x2_final
    if y1_final > y2_final: y1_final = y2_final

    roi = (x1_final, y1_final, x2_final, y2_final)
    
    # Recalcula os nós lógicos dentro da ROI FINAL
    nodes_roi = set([n for n in nodes if (x1_final <= n[0] <= x2_final and y1_final <= n[1] <= y2_final)])
    
    if not nodes_roi:
         # Isso só deve acontecer se a ROI inicial era inválida (p.ex. um ponto único)
        print("AVISO: BBox final não contém nós lógicos da malha. Usando ROI inicial.")
        nodes_roi = nodes_roi_inicial
        roi = roi_inicial
        x1_final, y1_final, x2_final, y2_final = roi

    # Atualiza as arestas e grids para o novo ROI final
    edges_set = set()
    for u in nodes_roi:
        for v in adj.get(u, []):
            if v in nodes_roi and u < v:
                edges_set.add(_norm_edge(u, v))

    # grids ordenados do ROI (para localizar células)
    grid_xs_final = sorted({n[0] for n in nodes_roi})
    grid_ys_final = sorted({n[1] for n in nodes_roi})
    nodes_set = set(nodes_roi)

    # print(f"ROI FINAL (BBox mínima): {roi} área={(x2_final - x1_final) * (y2_final - y1_final)}")


    # 5) ligar cada especial às **quatro** quinas (duas diagonais) da célula
    conexoes_especiais = []
    for tipo, p in pts_atribuidos:
        # Verifica se o ponto especial ainda está DENTRO OU NAS BORDAS do ROI final
        if not (x1_final <= p[0] <= x2_final and y1_final <= p[1] <= y2_final):
            # Não deveria acontecer se o BBox foi calculado corretamente
            continue
            
        conns = conectar_especial_por_diagonais(p, grid_xs_final, grid_ys_final, nodes_set, edges_set)
        if conns:
            conexoes_especiais.extend(conns)
        else:
            print(f"AVISO: {tipo} em {p} não conseguiu conectar por diagonais sem violar regras no ROI final.")

    # 6) grafo final e saídas
    edges_roi = list(edges_set)  # apenas arestas H/V entre nós lógicos
    # A função montar_grafo precisa usar o IDMAP dos nós LÓGICOS FINAIS
    grafo = montar_grafo(nodes_roi, edges_roi, pts_atribuidos, conexoes_especiais)

    img_mapa_out = os.path.join(output_dir, "grafo_sobre_mapa.png")
    img_branco_out = os.path.join(output_dir, "grafo_fundo_branco.png")
    grafo_txt = os.path.join(output_dir, "grafo.txt")

    img_resultado = desenhar_grafo_mapa(imagem_mapa, roi, grafo, img_mapa_out)
    desenhar_grafo_mapa(None, roi, grafo, img_branco_out)  # fundo branco
    salvar_grafo_txt(grafo, grafo_txt)

    print("\n=== SISTEMA CRIADO (especiais conectados por DIAGONAIS às quinas) ===")
    print(f"ROI selecionada: {roi} área={(x2_final - x1_final) * (y2_final - y1_final)}")
    print(f"Grafo sobre mapa: {img_mapa_out}")
    print(f"Grafo fundo branco: {img_branco_out}")
    print(f"Arquivo do grafo: {grafo_txt}")
    print(f"Total de nós: {len(grafo)}")

    return {
        "retangulo": roi,
        "grafo": grafo,
        "pontos_atribuidos": pts_atribuidos,
        "cruzamentos_no_retangulo": sorted(nodes_roi),
        "imagem_resultado": img_resultado,
    }

# ---------------------------
# Exemplo de uso direto
# ---------------------------
if __name__ == "__main__":
    N_VERTIPORT, N_ESTACAO, N_FORNECEDOR, N_CLIENTE = 1, 1, 2, 2

    res = criar_sistema_logistico(
        N_VERTIPORT,
        N_ESTACAO,
        N_FORNECEDOR,
        N_CLIENTE,
        arquivo_cruzamentos="pontos_cruzamentos.txt",
        arquivo_construcoes="pontos_construcoes.txt",
        imagem_mapa="mapafinal.png",
        output_dir="sistema_logistico",
        dist_minima_pontos=0,
        dist_minima_borda=500,
    )

    if res:
        print("\nOK! Visualizando…")
        plt.figure(figsize=(12, 5))
        if os.path.exists("mapafinal.png"):
            img_original = cv2.cvtColor(cv2.imread("mapafinal.png"), cv2.COLOR_BGR2RGB)
            plt.subplot(1, 2, 1)
            plt.imshow(img_original)
            plt.title("Mapa")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(res["imagem_resultado"])
            plt.title("Grafo")
            plt.axis("off")
        else:
            plt.imshow(res["imagem_resultado"])
            plt.title("Grafo")
            plt.axis("off")
        plt.tight_layout()
        plt.show()