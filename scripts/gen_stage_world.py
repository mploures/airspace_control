#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, math, argparse, shutil, random, subprocess
import cv2
import numpy as np

# -----------------------------
# Parse do grafo.txt
# -----------------------------

def compute_spawn_yaws(robot_poses_m, W_px, H_px, mode="radial", min_sep_deg=20):
    """
    Gera yaw distinto por robô.
    - radial: aponta do centro do mapa para fora (reduz confrontos frontais)
    - spread: uniformemente distribuídos em 2π
    - random: aleatório, com separação mínima
    """
    Cx, Cy = W_px / 2.0, H_px / 2.0
    N = len(robot_poses_m)
    yaws = []

    def sep_ok(th, used, eps):
        for u in used:
            d = abs((th - u + math.pi) % (2*math.pi) - math.pi)
            if d < eps:
                return False
        return True

    for i, (x, y) in enumerate(robot_poses_m):
        if mode == "spread":
            th = 2.0 * math.pi * i / max(1, N)
        elif mode == "random":
            th = random.uniform(-math.pi, math.pi)
        else:  # radial
            th = math.atan2(y - Cy, x - Cx)

        eps = math.radians(min_sep_deg)
        tries = 0
        while not sep_ok(th, yaws, eps) and tries < 72:
            th = ((th + eps + math.pi) % (2*math.pi)) - math.pi
            tries += 1

        yaws.append(th)
    return yaws


def read_grafo_txt(path):
    nodes = {}
    with open(path, 'r', encoding='utf-8') as f:
        _ = f.readline()  # cabeçalho
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^([^,]+),([^,]+),\(\s*([-\d]+)\s*,\s*([-\d]+)\s*\)(?:,(.*))?$', line)
            if not m:
                raise ValueError(f"Formato inválido: {line}")
            tipo, label, xs, ys, rest = m.groups()
            x, y = int(xs), int(ys)
            conns = [c.strip() for c in (rest.split(',') if rest else []) if c.strip()]
            nodes[label] = {'tipo': tipo, 'pos': (x, y), 'conns': conns}
    return nodes


def write_grafo_txt(path, nodes):
    """Escreve nodes no mesmo formato que lemos (com cabeçalho genérico)."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('tipo,label,(x,y),conexoes\n')
        for label, info in nodes.items():
            tipo = info.get('tipo', '')
            x, y = info.get('pos', (0, 0))
            conns = info.get('conns', [])
            if conns:
                f.write(f"{tipo},{label},({x},{y}),{','.join(conns)}\n")
            else:
                f.write(f"{tipo},{label},({x},{y})\n")


def get_nodes_bbox(nodes):
    xs, ys = [], []
    for info in nodes.values():
        x, y = info['pos']
        xs.append(x); ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def align_grafo_to_bitmap(nodes, W_img, H_img, out_path=None, force=False):
    """
    Alinha o grafo a um bitmap recortado aplicando transformação afim simples:
      x' = (x - minx) * sx
      y' = (y - miny) * sy
    Onde (minx,miny) é o offset do bbox dos nós e sx,sy são escalas para casar
    o bbox com as dimensões do bitmap. Casos:
      - Se o bbox ~ (W_img,H_img): aplica só OFFSET (sx=sy=1).
      - Se ratios W_img/w_nodes e H_img/h_nodes ~ iguais: aplica ESCALA UNIFORME.
      - Caso contrário, aplica ESCALA ANISOTRÓPICA (sx!=sy).
    """
    bbox = get_nodes_bbox(nodes)
    if not bbox:
        return nodes, (0, 0), (1.0, 1.0), None

    minx, miny, maxx, maxy = bbox
    w_nodes = maxx - minx + 1
    h_nodes = maxy - miny + 1

    tol_w = max(8, int(0.02 * W_img))
    tol_h = max(8, int(0.02 * H_img))

    rx = W_img / max(1, w_nodes)
    ry = H_img / max(1, h_nodes)

    same_size = (abs(w_nodes - W_img) <= tol_w) and (abs(h_nodes - H_img) <= tol_h)
    nearly_uniform = abs(rx - ry) <= 0.01

    sx = sy = 1.0
    # OBS: mesmo que force=True, se não for "same_size", iremos escalar.
    if same_size:
        sx = sy = 1.0
    elif nearly_uniform:
        sx = sy = (rx + ry) * 0.5
    else:
        sx, sy = rx, ry

    aligned = {}
    for label, info in nodes.items():
        x, y = info['pos']
        x2 = int(round((x - minx) * sx))
        y2 = int(round((y - miny) * sy))
        aligned[label] = {
            'tipo': info.get('tipo', ''),
            'pos': (x2, y2),
            'conns': list(info.get('conns', []))
        }

    out_written = None
    if out_path:
        try:
            write_grafo_txt(out_path, aligned)
            out_written = out_path
        except Exception as e:
            print(f"[WARN] Não consegui escrever grafo alinhado em {out_path}: {e}")

    print(f"[INFO] Alinhamento do grafo ao bitmap:"
          f" offset=({minx},{miny})  bbox=({w_nodes}x{h_nodes})"
          f" img=({W_img}x{H_img})  ratios=(rx={rx:.4f}, ry={ry:.4f})"
          f" -> escalas aplicadas (sx={sx:.4f}, sy={sy:.4f})")
    return aligned, (minx, miny), (sx, sy), out_written


def get_deposit_points_from_grafo(nodes):
    """
    Coleta pontos de depósito a partir do grafo.
    - Aceita DEPOSITO/DEPOSIT/DEP/DEPO (quando existem)
    - **E também VERTIPORT** (trata como “depósito” operacional/inicial).
    """
    deps = []
    for label, info in nodes.items():
        t = (info.get('tipo', '') or '').strip().upper()
        lbl = (label or '').upper()
        if (t in {'DEPOSITO', 'DEPOSIT', 'DEP', 'DEPO'} or
            'DEPOS' in t or 'DEPOS' in lbl or
            t == 'VERTIPORT'):
            deps.append(info['pos'])
    return deps


def make_graph_and_wall_bitmaps(original_path, worlds_dir, W, H, border_frac=0.003):
    """
    - Binariza: tudo que NÃO for branco -> PRETO; branco permanece.
    - Gera 'muro': imagem só com retângulo de borda em PRETO.
    Retorna: (graph_bw_name, muro_name, abs_graph_path, abs_muro_path)
    """
    base = os.path.splitext(os.path.basename(original_path))[0]
    graph_bw_name = f"{base}_bw.png"
    muro_name     = f"{base}_muro.png"
    graph_bw_path = os.path.join(worlds_dir, graph_bw_name)
    muro_path     = os.path.join(worlds_dir, muro_name)

    img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Não consegui ler {original_path} para binarizar.")

    if img.ndim == 2:
        gray = img
        mask_nonwhite = (gray < 250)
    else:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if img.shape[2] == 4:
            a = img[:, :, 3]
            mask_nonwhite = ((b < 250) | (g < 250) | (r < 250)) & (a > 0)
        else:
            mask_nonwhite = ((b < 250) | (g < 250) | (r < 250))

    graph = np.full((H, W), 255, dtype=np.uint8)
    graph[mask_nonwhite] = 0
    cv2.imwrite(graph_bw_path, graph)

    th = max(1, int(round(max(W, H) * border_frac)))
    muro = np.full((H, W), 255, dtype=np.uint8)
    muro[:th, :] = 0
    muro[-th:, :] = 0
    muro[:, :th] = 0
    muro[:, -th:] = 0
    cv2.imwrite(muro_path, muro)

    return graph_bw_name, muro_name, graph_bw_path, muro_path


def compute_scale(W, H, max_wh):
    if max_wh is None or max_wh <= 0:
        return 1.0
    return min(1.0, float(max_wh) / max(W, H))


def downscale_with_scale(src_path, worlds_dir, W, H, s):
    """Redimensiona src_path por fator s (<=1). Retorna (name, newW, newH)."""
    if s >= 0.9999:
        return os.path.basename(src_path), W, H
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Não consegui abrir {src_path} para downscale.")
    newW, newH = int(round(W * s)), int(round(H * s))
    out_name = f"{os.path.splitext(os.path.basename(src_path))[0]}_stage{int(round(s * 100))}.png"
    out_path = os.path.join(worlds_dir, out_name)
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, resized)
    return out_name, newW, newH


# -----------------------------
# Leitura robusta dos pontos_* (fallback se não houver grafo)
# -----------------------------
def read_points_txt(path):
    pts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.findall(r'-?\d+', line)
            if len(m) >= 2:
                x, y = int(m[0]), int(m[1])
                pts.append((x, y))
    return pts


# -----------------------------
# Geometria / conversão de coords
# -----------------------------
def to_stage_xy(px, py, W_px, H_px):
    """
    Converte (x_px, y_px) (origem no canto superior-esquerdo da imagem)
    para o Stage com floorplan em pose [W/2 H/2 ...]:
      - mantém X (sem espelho)
      - inverte Y: y_stage = H_px - py
    """
    return float(px), float(H_px - py)


def wrap_pi(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def angdiff(a, b):
    return abs(wrap_pi(a - b))


def r_bound_to_image(x0, y0, theta, W, H, sep):
    c, s = math.cos(theta), math.sin(theta)
    INF = 1e18
    rx = ry = INF
    if abs(c) > 1e-12:
        rx = (W - sep - x0) / c if c > 0 else (x0 - sep) / (-c)
    if abs(s) > 1e-12:
        ry = (H - sep - y0) / s if s > 0 else (y0 - sep) / (-s)
    r = min(rx, ry)
    return max(0.0, r)


# -----------------------------
# Paleta de cores (40+)
# -----------------------------
PALETTE = [
    "blue", "purple", "green", "orange", "red", "magenta", "cyan", "yellow",
    "brown", "navy", "maroon", "olive", "teal", "silver", "gray", "gold",
    "pink", "orchid", "chocolate", "turquoise", "violet", "salmon", "khaki",
    "coral", "crimson", "indigo", "lime", "plum", "tan", "wheat", "seagreen",
    "slateblue", "darkgreen", "darkorange", "deepskyblue", "hotpink",
    "darkviolet", "dodgerblue", "forestgreen", "fuchsia", "lightseagreen"
]
def color_for(i): return PALETTE[i % len(PALETTE)]


# -----------------------------
# Ângulos “bons” (entre arestas)
# -----------------------------
def gap_angles_from_segments(center, corners, need):
    x0, y0 = center
    if not corners:
        M = max(32, need * 4)
        base = random.random() * 2 * math.pi
        for k in range(M):
            yield wrap_pi(base + 2 * math.pi * k / M)
        return

    alphas = [math.atan2(cy - y0, cx - x0) for (cx, cy) in corners]
    alphas = sorted([wrap_pi(a) for a in alphas])
    gaps = []
    for i in range(len(alphas)):
        a1 = alphas[i]
        a2 = alphas[(i + 1) % len(alphas)]
        width = wrap_pi(a2 - a1)
        if width <= 0:
            width += 2 * math.pi
        gaps.append((a1, width))

    produced = 0
    round_k = 1
    while produced < need * 6:
        for (a_start, width) in gaps:
            for j in range(1, round_k + 1):
                theta = wrap_pi(a_start + (width * j) / (round_k + 1))
                yield theta
                produced += 1
        round_k += 1


def candidate_for_angle(center, corners, theta, sep, W, H, global_pts,
                        r_min=6, step_r=3, max_push=2000):
    x0, y0 = center
    alphas = [math.atan2(cy - y0, cx - x0) for (cx, cy) in corners]
    if alphas:
        delta = min(angdiff(theta, a) for a in alphas)
        if delta < 1e-3:
            delta = 1e-3
        r_needed = max(r_min, (sep / max(1e-6, math.sin(delta))))
    else:
        r_needed = r_min
    r_max_img = r_bound_to_image(x0, y0, theta, W, H, sep)
    if r_max_img < r_needed:
        return None

    r = r_needed
    pushes = 0
    while r <= r_max_img and pushes <= max_push:
        x = int(round(x0 + r * math.cos(theta)))
        y = int(round(y0 + r * math.sin(theta)))
        ok = True
        for (gx, gy) in global_pts:
            if math.hypot(x - gx, y - gy) < sep:
                ok = False
                break
        if ok:
            return (x, y)
        r += step_r
        pushes += 1
    return None


# -----------------------------
# .world do Stage (duas camadas; 1px = 1m)
# -----------------------------
def generate_world_text_layers(bitmap_graph_low, bitmap_muro_high,
                               W_px, H_px,
                               robot_poses_m, robot_size_m,
                               low_height=0.05, wall_height=2.0,
                               robot_z=1.0,
                               sense_walls=False,
                               collide_walls=False,
                               robot_yaws=None,
                               robot_height_m=0.30,          # <<< NOVO: altura do corpo do robô
                               lidar_pose_rel=None):          # <<< NOVO: (lx, ly, lz, lyaw)
    # Se não for informado, coloca o LIDAR no meio do corpo (Z = metade da altura)
    if lidar_pose_rel is None:
        lidar_pose_rel = (0.0, 0.0, robot_height_m/2.0, 0.0)

    lx, ly, lz, lyaw = lidar_pose_rel

    wall_W = W_px * 1.1
    wall_H = H_px * 1.05

    header = f'''define floorplan model (
  color "gray90"
  boundary 1
  gui_nose 0
  gui_grid 0
  gui_move 0
  gui_outline 0
  gripper_return 0
  fiducial_return 0
)

define topurg ranger
(
  # o "corpo" do sensor não colide nem reflete laser
  obstacle_return 0
  laser_return 0

  # o que o sensor enxerga
  sensor(
    range [ 0.01 2.0 ]
    fov 360
    samples 360
    obstacle_return 1
  )

  color "black"
  size [0.100 0.100 0.100]
)

define erratic position
(
  size [{robot_size_m:.2f} {robot_size_m:.2f} {robot_height_m:.2f}]
  # Robôs são visíveis uns aos outros (evitação)
  obstacle_return 1
  laser_return 1
  drive "diff"
  topurg(pose [ {lx:.3f} {ly:.3f} {lz:.3f} {lyaw:.3f} ])  # <<< LIDAR NO MEIO
)

window (
  size [700 700]
  rotate [0.000 0.000]
  scale {max(10, int(max(W_px, H_px)/10))}
  show_data 1
  show_blocks 1
  show_flags 0
  show_clock 1
  show_follow 0
  show_footprints 1
  show_grid 1
  show_status 1
  show_trailarrows 0
  show_occupancy 0
)

# --- camada baixa: grafo (visual) ---
floorplan
(
  name "graph_low"
  bitmap "{bitmap_graph_low}"
  color "black"
  size [{W_px:.3f} {H_px:.3f} {low_height:.3f}]
  pose [{W_px/2:.3f} {H_px/2:.3f} 0.0 0.0]
  obstacle_return 0
  laser_return 0
)

# --- camada alta: MURO (real/opcional) ---
floorplan
(
  name "muro_high"
  bitmap "{bitmap_muro_high}"
  color "black"
  size [{wall_W:.3f} {wall_H:.3f} {wall_height:.3f}]
  pose [{W_px/2:.3f} {H_px/2:.3f} 0.0 0.0]
  obstacle_return {1 if collide_walls else 0}
  laser_return {1 if sense_walls else 0}
)
'''
    body = []
    for i, (x, y) in enumerate(robot_poses_m):
        yaw = (robot_yaws[i] if (robot_yaws and i < len(robot_yaws)) else 0.0)
        body.append(
            f'erratic( pose [ {x:.3f} {y:.3f} {robot_z:.3f} {yaw:.3f} ] name "vant_{i}" color "{color_for(i)}")'
        )
    return header + "\n".join(body) + "\n"




# -----------------------------
# Paths ROS/pkg
# -----------------------------
def get_package_dir():
    try:
        import rospkg
        return rospkg.RosPack().get_path('airspace_control')
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def smart_find_assets(pkg_dir, grafo_arg, bitmap_arg):
    """
    Decide bitmap + (grafo OU pontos_*). Se faltar, roda processar_mapa.py e/ou selecionar_cluster.py.
    **Prioriza SEMPRE** gerar o grafo quando ele não existir.
    Retorna: (bitmap_path, grafo_path or None, points_path or None)
    """
    sys_dir  = os.path.join(pkg_dir, 'graph', 'sistema_logistico')
    flat_dir = os.path.join(pkg_dir, 'graph')

    def pick_bitmap():
        cands = []
        for base in (sys_dir, flat_dir):
            cands += [
                os.path.join(base, 'grafo_fundo_branco.png'),
                os.path.join(base, 'mapafinal.png'),
                os.path.join(base, 'mapa_completo.png'),
                os.path.join(base, 'map.png'),
            ]
        if bitmap_arg and os.path.isfile(bitmap_arg):
            return bitmap_arg
        for p in cands:
            if os.path.isfile(p):
                return p
        return None

    def pick_grafo():
        if grafo_arg and os.path.isfile(grafo_arg):
            return grafo_arg
        for base in (sys_dir, flat_dir):
            p = os.path.join(base, 'grafo.txt')
            if os.path.isfile(p):
                return p
        return None

    def pick_points():
        for base in (sys_dir, flat_dir):
            for name in ('pontos_cruzamentos.txt', 'pontos_construcoes.txt'):
                p = os.path.join(base, name)
                if os.path.isfile(p):
                    return p
        return None

    def ensure_points():
        pts = pick_points()
        if pts:
            return pts
        pm_path = os.path.join(flat_dir, 'processar_mapa.py')
        if os.path.isfile(pm_path):
            print("[INFO] Gerando pontos base com graph/processar_mapa.py ...")
            try:
                subprocess.run(['python3', pm_path], cwd=flat_dir, check=True)
            except subprocess.CalledProcessError as e:
                print("[WARN] processar_mapa.py retornou código", e.returncode)
        return pick_points()

    def try_build_grafo():
        sc_path = os.path.join(flat_dir, 'selecionar_cluster.py')
        if not os.path.isfile(sc_path):
            return None
        print("[INFO] Gerando grafo com graph/selecionar_cluster.py ...")
        try:
            subprocess.run(['python3', sc_path], cwd=flat_dir, check=True)
        except subprocess.CalledProcessError as e:
            print("[WARN] selecionar_cluster.py retornou código", e.returncode)
        return pick_grafo()

    bmp = pick_bitmap()
    grf = pick_grafo()
    pts = pick_points()

    if not grf:
        if not pts:
            pts = ensure_points()
        grf = try_build_grafo()
        bmp_grafo = os.path.join(sys_dir, 'grafo_fundo_branco.png')
        if os.path.isfile(bmp_grafo):
            bmp = bmp_grafo
        elif not bmp:
            bmp = pick_bitmap()

    if not bmp:
        pts = ensure_points()
        bmp = pick_bitmap()

    if bmp and (grf or pts):
        return bmp, grf, pts

    raise FileNotFoundError(
        "Não encontrei bitmap e fontes (grafo.txt ou pontos_*.txt) mesmo após tentar regenerar. "
        "Verifique se há alguma imagem base em <pkg>/graph e se selecionar_cluster.py/ processar_mapa.py estão OK."
    )


# -----------------------------
# MAIN
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grafo',  default=None, help='grafo.txt (opcional)')
    ap.add_argument('--bitmap', default=None, help='bitmap base (opcional)')
    ap.add_argument('--out',    default=None, help='padrão: <pkg>/worlds/airspace.world')
    ap.add_argument('--nvants', type=int, default=3, help='total de VANTs')
    ap.add_argument('--sep_px', type=int, default=5, help='separação mínima entre VANTs (px) (usado só se faltar depósito)')
    ap.add_argument('--robot_size_m', type=float, default=1.2, help='tamanho visual no Stage (m)')
    ap.add_argument('--max_wh', type=int, default=200,
                    help='limite máx. de pixels para largura/altura do bitmap no Stage (downscale automático); 0 desativa')

    args = ap.parse_args()

    if args.nvants <= 0:
        raise RuntimeError("nvants deve ser >= 1")
    random.seed(42); np.random.seed(42)

    pkg_dir = get_package_dir()
    worlds_dir = os.path.join(pkg_dir, 'worlds')
    os.makedirs(worlds_dir, exist_ok=True)

    # 1) Decide assets (gera se necessário)
    bitmap_path, grafo_path, points_path = smart_find_assets(pkg_dir, args.grafo, args.bitmap)

    # 2) Carrega bitmap (dimensão = mundo original)
    img = cv2.imread(bitmap_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Não consegui abrir {bitmap_path}")
    H, W = img.shape[:2]

    # 3) Garante bitmap original dentro de worlds/
    bitmap_basename = os.path.basename(bitmap_path)
    bitmap_target = os.path.join(worlds_dir, bitmap_basename)
    if os.path.abspath(bitmap_path) != os.path.abspath(bitmap_target):
        shutil.copy2(bitmap_path, bitmap_target)

    # 4) Binariza grafo (não-branco -> preto) e gera MURO (borda preta)
    graph_bw_name, muro_name, graph_bw_path, muro_path = make_graph_and_wall_bitmaps(
        bitmap_target, worlds_dir, W, H, border_frac=0.003
    )

    # 5) Lê e ALINHA o grafo ao bitmap (migração padronizada de TODAS as posições)
    nodes = None
    offset_used  = (0, 0)
    scales_used  = (1.0, 1.0)
    grafo_out_path = None
    if grafo_path:
        nodes_in = read_grafo_txt(grafo_path)
        # NÃO forçar offset-only; deixa escalar quando necessário
        force_align = False
        out_norm_path = os.path.join(os.path.dirname(grafo_path), 'grafo_recortado.txt')
        nodes_aligned, offset_used, scales_used, grafo_out_path = align_grafo_to_bitmap(
            nodes_in, W_img=W, H_img=H, out_path=out_norm_path, force=force_align
        )
        nodes = nodes_aligned
    else:
        nodes = None

    # 6) Posições: exatamente os "depósitos" (VERTIPORT quando não houver DEPOSITO).
    
    placed_px = []
    if nodes:
        deposits = get_deposit_points_from_grafo(nodes)
        if not deposits:
            print("[WARN] grafo.txt sem depósitos/VERTIPORT identificados; nada a posicionar.")
        else:
            if len(deposits) < args.nvants:
                print(f"[WARN] Só {len(deposits)} pontos-base para {args.nvants} VANTs. "
                      f"Distribuindo ciclicamente os VANTs ao redor dos mesmos pontos.")

            n_vants = args.nvants
            n_ports = len(deposits)

            # número balanceado de VANTs por VERTIPORT
            base_per_port = n_vants // n_ports
            extra = n_vants % n_ports
            distribution = [base_per_port + (1 if i < extra else 0) for i in range(n_ports)]

            # distância física mínima entre centros
            robot_d = args.robot_size_m
            safety_margin = args.sep_px
            min_sep = robot_d + safety_margin

            for i, (cx, cy) in enumerate(deposits):
                cx, cy = map(int, (cx, cy))
                n_here = distribution[i]
                if n_here == 0:
                    continue

                # o primeiro VANT fica exatamente sobre o ponto do VERTIPORT
                placed_px.append((cx, cy))

                # define raio base de afastamento em função do tamanho físico
                radius = min_sep
                camada = 1

                for j in range(1, n_here):
                    # calcula ângulo de distribuição equilibrada (uniforme em 360°)
                    angle = 2 * math.pi * (j - 1) / max(1, n_here - 1)

                    # coordenadas candidatas
                    x_new = int(round(cx + radius * math.cos(angle)))
                    y_new = int(round(cy + radius * math.sin(angle)))

                    # checa colisão global
                    ok = False
                    attempt = 0
                    while not ok and attempt < 20:
                        ok = True
                        for (px, py) in placed_px:
                            if math.hypot(px - x_new, py - y_new) < min_sep:
                                ok = False
                                break
                        if not ok:
                            # aumenta o raio e tenta novamente
                            camada += 1
                            radius = (robot_d + safety_margin) * camada
                            x_new = int(round(cx + radius * math.cos(angle)))
                            y_new = int(round(cy + radius * math.sin(angle)))
                        attempt += 1

                    placed_px.append((x_new, y_new))



    if not placed_px:
        # fallback (sem pontos-base)
        sep = max(1, int(args.sep_px))
        vports = []
        if nodes:
            for label, info in nodes.items():
                if (info.get('tipo', '') or '').upper() == 'VERTIPORT':
                    center = info['pos']
                    vports.append({'label': label, 'center': center, 'corners': []})
        if not vports:
            if not points_path:
                raise RuntimeError("Sem pontos-base (VERTIPORT/DEPOSITO) e sem pontos_* para fallback.")
            pts = read_points_txt(points_path)
            if not pts:
                raise RuntimeError(f"Arquivo de pontos vazio: {points_path}")
            vports = [{'label': f'VP_{i}', 'center': p, 'corners': []} for i, p in enumerate(pts)]

        M = len(vports)
        q, r = divmod(args.nvants, M)
        targets = [q + (1 if i < r else 0) for i in range(M)]
        assigned = [0] * M
        angle_iters = [gap_angles_from_segments(vp['center'], vp['corners'], targets[i] or 1)
                       for i, vp in enumerate(vports)]
        r_min = max(sep + 1, 6)
        step_r = max(2, int(sep))
        placed_count = 0
        while placed_count < args.nvants:
            progressed = False
            for i, vp in enumerate(vports):
                if assigned[i] >= targets[i]:
                    continue
                placed_here = False
                tries = 0
                while tries < 3000 and not placed_here:
                    try:
                        theta = next(angle_iters[i])
                    except StopIteration:
                        angle_iters[i] = gap_angles_from_segments(
                            vp['center'], vp['corners'],
                            targets[i] - assigned[i] + 1)
                        theta = next(angle_iters[i])
                    cand = candidate_for_angle(
                        center=vp['center'], corners=vp['corners'],
                        theta=theta, sep=sep, W=W, H=H, global_pts=placed_px,
                        r_min=r_min, step_r=step_r, max_push=4000)
                    tries += 1
                    if cand is None:
                        continue
                    placed_px.append(cand)
                    assigned[i] += 1
                    placed_count += 1
                    placed_here = True
                    progressed = True
                if placed_here and placed_count >= args.nvants:
                    break
            if not progressed:
                step_r = max(1, step_r - 1)
                for k in range(len(angle_iters)):
                    angle_iters[k] = gap_angles_from_segments(
                        vports[k]['center'], vports[k]['corners'],
                        targets[k] - assigned[k] + 2)

    # 7) Downscale PAREADO (mesmo 'scale' para grafo e muro)
    s = compute_scale(W, H, args.max_wh)
    graph_name_used, W_used, H_used = downscale_with_scale(graph_bw_path, worlds_dir, W, H, s)
    muro_name_used,  W2,     H2     = downscale_with_scale(muro_path,     worlds_dir, W, H, s)
    assert (W_used, H_used) == (W2, H2), "downscale inconsistente entre grafo e muro"

    # 8) Ajuste das poses para o tamanho usado (apenas inverte Y)
    if s != 1.0:
        robot_poses_m = [to_stage_xy(x * s, y * s, W_used, H_used) for (x, y) in placed_px]
    else:
        robot_poses_m = [to_stage_xy(x, y, W_used, H_used) for (x, y) in placed_px]

    z = 1.0  # altitude absoluta do robô no Stage
    robot_yaws = compute_spawn_yaws(robot_poses_m, W_used, H_used, mode="radial", min_sep_deg=12.0)

    world_text = generate_world_text_layers(
        bitmap_graph_low=graph_name_used,
        bitmap_muro_high=muro_name_used,
        W_px=W_used, H_px=H_used,
        robot_poses_m=robot_poses_m,
        robot_size_m=args.robot_size_m,
        low_height=0.05,
        wall_height=2.00,
        robot_z=z,
        sense_walls=True,
        collide_walls=True,
        robot_yaws=robot_yaws,
        robot_height_m=0.30,             # <<< mantém altura do corpo usada no world
        lidar_pose_rel=(0.0, 0.0, -0.15, 0.0)  # <<< LIDAR exatamente no meio do corpo
    )

    out_path = args.out or os.path.join(worlds_dir, 'airspace.world')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(world_text)

    # 10) Logs
    print(f"[OK] Gerado {out_path}")
    print(f"  - bitmap grafo: {os.path.join(worlds_dir, graph_name_used)}")
    print(f"  - bitmap muro : {os.path.join(worlds_dir, muro_name_used)}")
    if grafo_out_path:
        print(f"  - grafo alinhado salvo em: {grafo_out_path} (offset={offset_used}, scales={scales_used})")
    else:
        print(f"  - alinhamento: offset={offset_used} scales={scales_used} (0,0 e 1,1 se não aplicados)")
    print(f"  - size: [{W_used:.2f} {H_used:.2f}] m  resolution=1 px/m  (downscale s={s:.3f})")
    print(f"  - VANTs: {len(robot_poses_m)}/{args.nvants}")
    for i, (x, y) in enumerate(robot_poses_m):
        print(f"    vant_{i}: pose [{x:.2f} {y:.2f} {z:.2f} 0]  (z={z:.2f})  color={color_for(i)}")



if __name__ == '__main__':
    main()
