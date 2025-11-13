#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, math, random, threading
import rospy, rospkg
from nav_msgs.msg import Odometry
from airspace_control.srv import GotoXY, GotoXYRequest

# ---- tenta importar a classe VANT do seu arquivo uav_agent.py ----
try:
    from airspace_core.uav_agent import VANT
except Exception as _e:
    import importlib.util
    rp = rospkg.RosPack().get_path('airspace_control')
    _path = os.path.join(rp, 'airspace_core', 'uav_agent.py')
    spec = importlib.util.spec_from_file_location("uav_agent", _path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    VANT = m.VANT

def wait_world_ready(param_key="/airspace/world_path", timeout=20.0, poll=0.2):
    """Espera o rosparam com o caminho do .world e o arquivo existir em disco."""
    import os
    t0 = time.time()
    world_path = None
    while (time.time() - t0) < timeout and not rospy.is_shutdown():
        try:
            world_path = rospy.get_param(param_key, "")
            if world_path and os.path.isfile(world_path):
                return world_path
        except Exception:
            pass
        time.sleep(poll)
    return world_path  # pode ser "" se não encontrou


# ---------------- util ----------------
def parse_world_rect(world_path, layer_name="graph_low"):
    """
    Lê o retângulo útil de uma camada floorplan do .world (por ex. graph_low):
      floorplan ( name "graph_low" ... size [W H Z] ... pose [Cx Cy ...] )
    Retorna (xmin, xmax, ymin, ymax). Se não achar, cai em (0,200, 0,200).
    """
    try:
        with open(world_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        # Bloco da camada
        m1 = re.search(r'floorplan\s*\([^)]*name\s*"' + re.escape(layer_name) + r'"[^)]*\)', txt, flags=re.S|re.I)
        blob = m1.group(0) if m1 else txt

        m_size = re.search(r'size\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+[0-9.]+\s*\]', blob, flags=re.I)
        m_pose = re.search(r'pose\s*\[\s*([0-9.\-]+)\s+([0-9.\-]+)\s+[0-9.\-]+\s+[0-9.\-]+\s*\]', blob, flags=re.I)

        if m_size:
            W = float(m_size.group(1))
            H = float(m_size.group(2))
        else:
            W = H = None

        if m_pose:
            Cx = float(m_pose.group(1))
            Cy = float(m_pose.group(2))
        else:
            Cx = Cy = None

        if (W is not None) and (H is not None):
            # Se pose não vier, assume padrão [W/2 H/2 ...]
            if (Cx is None) or (Cy is None):
                Cx, Cy = W/2.0, H/2.0
            xmin, xmax = Cx - W/2.0, Cx + W/2.0
            ymin, ymax = Cy - H/2.0, Cy + H/2.0
            return (xmin, xmax, ymin, ymax)

    except Exception as e:
        rospy.logwarn(f"[manager] Falha ao ler world: {e}")

    rospy.logwarn("[manager] Não encontrei retângulo do world; usando (0,200)×(0,200).")
    return (0.0, 200.0, 0.0, 200.0)


class OdomTracker:
    """Assina a odometria de um robô e guarda (x,y)."""
    def __init__(self, name):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.ready = False
        self._lock = threading.Lock()
        # Principal
        self._sub1 = rospy.Subscriber(f"/{name}/odom", Odometry, self._cb, queue_size=10)
        # Fallback para setups que emitam base_pose_ground_truth
        self._sub2 = rospy.Subscriber(f"/{name}/base_pose_ground_truth", Odometry, self._cb, queue_size=10)

    def _cb(self, msg):
        with self._lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.ready = True

    def dist_to(self, xy):
        with self._lock:
            return math.hypot(self.x - xy[0], self.y - xy[1])


def clamp_goal(x, y, rect, margin):
    xmin, xmax, ymin, ymax = rect
    x = min(max(x, xmin + margin), xmax - margin)
    y = min(max(y, ymin + margin), ymax - margin)
    return (x, y)


def sample_goals_for_vant(i, rect, margin, n_goals=3):
    """
    3 metas "espalhadas" usando colunas [20%, 50%, 80%] em X e
    linhas alternadas em Y; tudo clampado ao retângulo útil (graph_low).
    """
    xmin, xmax, ymin, ymax = rect
    W = xmax - xmin
    H = ymax - ymin

    cols = [0.20, 0.50, 0.80]
    rows = [0.20, 0.35, 0.50, 0.65, 0.80]  # faixas horizontais
    y_idx = i % len(rows)

    goals = []
    for k in range(n_goals):
        x = xmin + cols[k % len(cols)] * W
        y = ymin + rows[(y_idx + k) % len(rows)] * H
        # jitter leve
        x += random.uniform(-0.05, 0.05) * W
        y += random.uniform(-0.05, 0.05) * H
        goals.append(clamp_goal(x, y, rect, margin))
    return goals


def start_vant_threads(vants):
    """Inicia cada VANT.spin() em uma thread daemon."""
    for v in vants:
        t = threading.Thread(target=v.spin, daemon=True)
        t.start()


def mission_thread(name, goals, tracker, goal_eps):
    """Envia 3 metas sequenciais via serviço GotoXY e espera atingir (pela Odom)."""
    srv_name = f"/{name}/goto_xy"
    rospy.loginfo(f"[manager] Aguardando serviço {srv_name} ...")
    rospy.wait_for_service(srv_name)
    goto = rospy.ServiceProxy(srv_name, GotoXY)

    # espera odom inicial
    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown() and not tracker.ready and (rospy.Time.now().to_sec() - t0 < 10.0):
        rospy.sleep(0.1)

    for (gx, gy) in goals:
        if rospy.is_shutdown(): break
        try:
            resp = goto(GotoXYRequest(x=gx, y=gy))
            rospy.loginfo(f"[{name}] goto({gx:.1f},{gy:.1f}) -> {resp.accepted} {resp.message}")
        except rospy.ServiceException as e:
            rospy.logwarn(f"[{name}] Falha GotoXY: {e}")
            continue

        # monitora distância até chegar
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            d = tracker.dist_to((gx, gy))
            if d < goal_eps:
                rospy.loginfo(f"[{name}] meta alcançada (d={d:.2f} < {goal_eps:.2f})")
                break
            r.sleep()


def main():
    rospy.init_node("multi_vant_manager")
    random.seed(42)

    nvants = int(rospy.get_param("~nvants", 3))
    goal_margin = float(rospy.get_param("~goal_margin", 3.0))
    goal_eps = float(rospy.get_param("~goal_tolerance", 0.8))  # deve casar com VANT.goal_eps

    world_path = rospy.get_param("/airspace/world_path", default="")
    if not world_path:
        rp = rospkg.RosPack().get_path('airspace_control')
        world_path = os.path.join(rp, "worlds", "airspace.world")

    # Retângulo útil baseado em graph_low (tudo acontece dentro dele)
    rect = parse_world_rect(world_path, layer_name="graph_low")
    xmin, xmax, ymin, ymax = rect
    rospy.loginfo(f"[manager] world: {world_path}  rect=([{xmin:.1f},{xmax:.1f}] x [{ymin:.1f},{ymax:.1f}])")

    # 1) Instancia N objetos VANT (mesmo nó, topics /vant_i/*)
    vants = [VANT(name=f"vant_{i}") for i in range(nvants)]
    start_vant_threads(vants)

    # 2) Trackers de Odom
    trackers = [OdomTracker(f"vant_{i}") for i in range(nvants)]

    # 3) Gera 3 metas por VANT dentro do retângulo (com margem)
    all_goals = [sample_goals_for_vant(i, rect, goal_margin, n_goals=3) for i in range(nvants)]
    for i, glist in enumerate(all_goals):
        rospy.loginfo(f"[manager] vant_{i} metas: {['(%.1f,%.1f)'%(x,y) for (x,y) in glist]}")

    # 4) Dispara missões (GotoXY)
    missions = []
    for i in range(nvants):
        th = threading.Thread(
            target=mission_thread,
            args=(f"vant_{i}", all_goals[i], trackers[i], goal_eps),
            daemon=True
        )
        th.start()
        missions.append(th)

    rospy.loginfo("[manager] Central pronta. Acompanhando missões...")
    rospy.spin()


if __name__ == "__main__":
    main()
