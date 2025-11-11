#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import re
import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from airspace_control.srv import (
    GetBattery, GetBatteryResponse,
    GotoXY, GotoXYRequest, GotoXYResponse
)

# ---------------------------------- Utils -----------------------------------
def sat(v, vmin, vmax): return max(vmin, min(vmax, v))

def wrap_pi(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _published_topics_snapshot():
    try:
        return {t for (t, _typ) in rospy.get_published_topics()}
    except Exception:
        return set()

def _pick_first_existing(cands, timeout=3.0, poll_dt=0.1):
    t0 = time.time()
    while (time.time() - t0) < timeout and not rospy.is_shutdown():
        existing = _published_topics_snapshot()
        for t in cands:
            if t in existing: return t
        time.sleep(poll_dt)
    existing = _published_topics_snapshot()
    for t in cands:
        if t in existing: return t
    return None

def _extract_index_from_name(name):
    m = re.search(r'_(\d+)$', str(name))
    return int(m.group(1)) if m else 0

# ---------------------------------- VANT ------------------------------------
class VANT:
    def __init__(self, name: str):
        self.name = name

        # Estado
        self.x = self.y = self.th = 0.0
        self.have_pose = False

        # LIDAR
        self.scan = None
        self._min_range = float("inf")
        self._min_bearing = 0.0

        # Objetivo
        self.goal = None
        self.goal_eps = rospy.get_param("~goal_tolerance", 0.8)

        # Ganhos/limites
        self.Kp = rospy.get_param("~Kp", 1.2)
        self.d = rospy.get_param("~d", 0.80)
        self.v_max = rospy.get_param("~v_max", 3.0)
        self.omega_max = rospy.get_param("~omega_max", 1.5)

        # Campos potenciais
        self.safe_range_lidar = rospy.get_param("~safe_range_lidar", 2.5)
        self.k_rep_lidar      = rospy.get_param("~k_rep_lidar", 2.0)
        self.k_att            = rospy.get_param("~k_att", 1.5)
        self.gamma_tan        = rospy.get_param("~gamma_tan", 1.2)
        self.k_damp           = rospy.get_param("~k_damp", 0.4)

        # Parâmetros de segurança (CBF + multi)
        self.robot_radius     = rospy.get_param("~robot_radius", 0.30)
        self.hard_stop_range  = rospy.get_param("~hard_stop_range", 0.80)   # LIDAR
        self.cbf_margin       = rospy.get_param("~cbf_margin", 0.20)
        self.alpha_cbf        = rospy.get_param("~alpha_cbf", 2.2)
        self.front_bias_deg   = rospy.get_param("~front_bias_deg", 120)
        self.ray_clip         = rospy.get_param("~ray_clip", 12.0)
        self.v_back           = rospy.get_param("~v_back", 0.30)
        self.rear_cone_deg    = rospy.get_param("~rear_cone_deg", 140)      # *** NOVO: cone traseiro p/ CBF LIDAR

        # CBF pareado (odom-odom)
        self.pair_margin      = rospy.get_param("~pair_margin", 0.40)       # folga sobre 2*raio
        self.alpha_pair       = rospy.get_param("~alpha_pair", 3.0)         # classe-K para pares
        self.side_escape_gain = rospy.get_param("~side_escape_gain", 0.7)
        self.pair_timeout_s   = rospy.get_param("~pair_timeout_s", 1.0)

        # Partida segura
        self.t_start          = time.time()
        self.start_stagger_s  = rospy.get_param("~start_stagger_s", 1.0)    # escadinha por índice
        self.speed_ramp_s     = rospy.get_param("~speed_ramp_s", 4.0)       # rampa até v_max

        # Bateria (simples)
        self.voltage_nom = rospy.get_param("~voltage_nom", 22.2)
        self.capacity_Wh = rospy.get_param("~capacity_Wh", 150.0)
        self.soc = 1.0
        self.i_base = rospy.get_param("~i_base", 2.0)
        self.i_vgain = rospy.get_param("~i_vgain", 3.0)
        self.i_wgain = rospy.get_param("~i_wgain", 2.0)
        self.last_batt_ts = time.time()

        # Autodetecção de tópicos
        idx = _extract_index_from_name(self.name)
        self.self_idx = idx
        ns_logical = f"/{self.name}"
        ns_robot   = f"/robot_{idx}"
        odom_candidates = [f"{ns_logical}/odom", f"{ns_logical}/base_pose_ground_truth",
                           f"{ns_robot}/odom",   f"{ns_robot}/base_pose_ground_truth"]
        scan_candidates = [f"{ns_logical}/base_scan", f"{ns_robot}/base_scan"]
        cmd_candidates  = [f"{ns_robot}/cmd_vel", f"{ns_logical}/cmd_vel"]

        odom_topic = _pick_first_existing(odom_candidates, timeout=5.0) or odom_candidates[-1]
        scan_topic = _pick_first_existing(scan_candidates, timeout=5.0) or scan_candidates[-1]
        cmd_sim_topic = cmd_candidates[0]

        self._odom_topic_used = odom_topic
        self._scan_topic_used = scan_topic
        self._cmd_sim_topic   = cmd_sim_topic

        self.pub_cmd_logical = rospy.Publisher(f"{ns_logical}/cmd_vel", Twist, queue_size=10)
        self.pub_cmd_sim     = rospy.Publisher(cmd_sim_topic, Twist, queue_size=10)

        rospy.Subscriber(odom_topic, Odometry, self._cb_odom, queue_size=10)
        rospy.Subscriber(scan_topic, LaserScan, self._cb_scan, queue_size=10)

        # Services
        self.srv_batt = rospy.Service(f"{ns_logical}/get_battery", GetBattery, self._srv_get_battery)
        self.srv_goto = rospy.Service(f"{ns_logical}/goto_xy",     GotoXY,    self._srv_goto_xy)

        # Descoberta de vizinhos (assina odom dos outros)
        self.neigh = {}  # k -> dict(x,y,th,ts,has)
        self._setup_neighbors()

        self.rate = rospy.Rate(rospy.get_param("~rate_hz", 30.0))
        rospy.loginfo(f"[{self.name}] pronto. LIDAR + PairCBF SafetyShield "
                      f"(odom='{odom_topic}', scan='{scan_topic}', cmd_sim='{cmd_sim_topic}')")

    # ------------------------ Descoberta de vizinhos -------------------------
    def _setup_neighbors(self):
        topics = _published_topics_snapshot()
        def _has_topic(t): return t in topics
        indices = set()
        for t in topics:
            m = re.search(r'/robot_(\d+)/odom', t)
            if m: indices.add(int(m.group(1)))
        if not indices:
            for t in topics:
                m = re.search(r'/vant_(\d+)/(?:odom|base_pose_ground_truth)', t)
                if m: indices.add(int(m.group(1)))

        for k in sorted(indices):
            if k == self.self_idx: 
                continue
            cands = [f"/robot_{k}/odom", f"/vant_{k}/odom", f"/vant_{k}/base_pose_ground_truth"]
            topic_k = next((c for c in cands if _has_topic(c)), None)
            if topic_k is None: continue
            self.neigh[k] = {"x":0.0, "y":0.0, "th":0.0, "ts":0.0, "has":False}
            rospy.Subscriber(topic_k, Odometry, self._mk_cb_neigh(k), queue_size=5)
        if self.neigh:
            rospy.loginfo(f"[{self.name}] vizinhos detectados: {sorted(self.neigh.keys())}")

    def _mk_cb_neigh(self, k):
        def _cb(msg: Odometry):
            p = self.neigh[k]
            p["x"]  = msg.pose.pose.position.x
            p["y"]  = msg.pose.pose.position.y
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            siny_cosp = 2 * (qw*qz + qx*qy)
            cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
            p["th"] = math.atan2(siny_cosp, cosy_cosp)
            p["ts"] = time.time()
            p["has"] = True
        return _cb

    # ----------------------------- Callbacks ---------------------------------
    def _cb_odom(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        self.th = math.atan2(siny_cosp, cosy_cosp)
        self.have_pose = True

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg
        self._min_range = float("inf")
        self._min_bearing = 0.0
        ang = msg.angle_min
        inc = msg.angle_increment
        for r in msg.ranges:
            if not math.isfinite(r): r = self.ray_clip
            r = min(max(r, 0.0), self.ray_clip)
            if r < self._min_range:
                self._min_range  = r
                self._min_bearing = ang
            ang += inc

    # ------------------------------ Services ---------------------------------
    def _srv_get_battery(self, _req):
        self._battery_update(0.0, 0.0)
        rem_min = self._estimate_remaining_minutes()
        return GetBatteryResponse(
            soc=self.soc, voltage=self.voltage_nom,
            current=getattr(self, "_last_current", 0.0),
            remaining_minutes=rem_min,
        )

    def _srv_goto_xy(self, req: GotoXYRequest):
        self.goal = (float(req.x), float(req.y))
        rospy.loginfo(f"[{self.name}] Novo objetivo: ({req.x:.2f}, {req.y:.2f})")
        return GotoXYResponse(accepted=True, message="Objetivo recebido")

    # ------------------------------ Bateria ----------------------------------
    def _battery_update(self, v_cmd, w_cmd):
        now = time.time()
        dt = max(1e-3, now - self.last_batt_ts)
        self.last_batt_ts = now
        i = self.i_base + self.i_vgain * abs(v_cmd) + self.i_wgain * abs(w_cmd)
        self._last_current = i
        wh = (i * self.voltage_nom) * (dt / 3600.0)
        d_soc = wh / self.capacity_Wh
        self.soc = max(0.0, self.soc - d_soc)

    def _estimate_remaining_minutes(self):
        i_est = self.i_base + self.i_vgain * (0.5 * self.v_max) + self.i_wgain * (0.5 * self.omega_max)
        p_est = i_est * self.voltage_nom
        e_rem_Wh = self.soc * self.capacity_Wh
        if p_est <= 1e-6: return float("inf")
        return max(0.0, 60.0 * (e_rem_Wh / p_est))

    # ----------------------- Campos + Tangencial ------------------------------
    def _repulsion_and_tangent_from_lidar(self):
        if self.scan is None: return 0.0, 0.0
        u_rx = u_ry = u_tx = u_ty = 0.0
        angle = self.scan.angle_min
        inc   = self.scan.angle_increment
        front_h = math.radians(self.front_bias_deg) * 0.5

        if self.goal is not None:
            ex, ey = self.goal[0] - self.x, self.goal[1] - self.y
        else:
            ex = ey = 0.0

        for r in self.scan.ranges:
            if not math.isfinite(r): r = self.ray_clip
            r = min(max(r, 0.0), self.ray_clip)
            if 0.05 < r < self.safe_range_lidar:
                gth = self.th + angle
                dx, dy = math.cos(gth), math.sin(gth)
                front_w = 1.5 if abs(angle) <= front_h else 1.0
                rep = self.k_rep_lidar * front_w * max(0.0, (1.0 / max(0.1, r) - 1.0 / self.safe_range_lidar))
                rep *= 1.0 / max(0.1, r * r)
                # Repulsão
                u_rx -= rep * dx
                u_ry -= rep * dy
                # Tangencial
                cross_z = ex * dy - ey * dx
                sgn = 1.0 if cross_z >= 0.0 else -1.0
                u_tx += self.gamma_tan * sgn * (-dy) * rep
                u_ty += self.gamma_tan * sgn * ( dx) * rep
            angle += inc

        return (u_rx + u_tx, u_ry + u_ty)

    def _attractive_to_goal(self):
        if self.goal is None: return 0.0, 0.0, 0.0
        ex, ey = self.goal[0] - self.x, self.goal[1] - self.y
        dist = math.hypot(ex, ey)
        if dist < self.goal_eps: return 0.0, 0.0, dist
        return self.k_att * ex, self.k_att * ey, dist

    # --------------------- CBF LIDAR + CBF Pareado (multi) -------------------
    def _cbf_project_multi(self, v_des, w_des):
        """
        Projeta (v_des, w_des) no conjunto seguro, combinando:
          1) Barreiras LIDAR para feixes à frente (limite superior de v) e atrás (limite inferior de v)
          2) Barreiras pareadas (odom-odom) para cada vizinho j, também BIDIRECIONAIS
        Retorna: (v_cmd, w_cmd)
        """
        # Limites (superior/inferior) da velocidade linear
        v_upper = self.v_max
        v_lower = -self.v_max

        # -------- 1) LIDAR (bidirecional) --------
        if self.scan is not None:
            alpha  = self.alpha_cbf
            r_safe = self.hard_stop_range + self.cbf_margin

            angle = self.scan.angle_min
            inc   = self.scan.angle_increment
            for r in self.scan.ranges:
                if not math.isfinite(r): r = self.ray_clip
                r = min(max(r, 0.0), self.ray_clip)

                phi = angle
                c   = math.cos(phi)
                h   = r - r_safe  # função de barreira do feixe

                # À frente (c>0): v <= alpha * h / c
                if c > 1e-3:
                    v_upper = min(v_upper, alpha * h / c)
                # Atrás (c<0):  v >= alpha * h / c   (RHS tende a ser negativo)
                elif c < -1e-3:
                    v_lower = max(v_lower, alpha * h / c)

                angle += inc

            # Escape angular se algo estiver muito perto em QUALQUER direção
            if self._min_range < (r_safe + 0.30):
                turn_dir = -1.0 if self._min_bearing > 0 else 1.0
                w_des = sat(w_des + 0.7 * self.omega_max * turn_dir, -self.omega_max, self.omega_max)

        # -------- 2) CBF PAREADO (odom-odom), bidirecional --------
        R_pair = 2.0 * self.robot_radius + self.pair_margin
        now = time.time()
        for k, p in self.neigh.items():
            if not p["has"] or (now - p["ts"]) > 1.0:
                continue

            dx = p["x"] - self.x
            dy = p["y"] - self.y
            d  = math.hypot(dx, dy)
            if d <= 1e-6:
                continue

            phi = wrap_pi(math.atan2(dy, dx) - self.th)
            c   = math.cos(phi)
            h   = d - R_pair  # barreira pareada

            # Hard stop e "abrir" quando MUITO perto
            if d < (R_pair + 0.15):
                if abs(phi) < math.radians(100):
                    # Vizinho essencialmente à frente: limita avanço, permite ré suave
                    v_upper = min(v_upper, 0.10)
                    v_lower = max(v_lower, -self.v_back)
                else:
                    # Vizinho essencialmente atrás: permite ir pra frente suave, proíbe ré
                    v_upper = min(v_upper, self.v_back)
                    v_lower = max(v_lower, 0.0)

                turn_dir = -1.0 if phi > 0.0 else 1.0
                w_des = sat(w_des + self.side_escape_gain * self.omega_max * turn_dir,
                            -self.omega_max, self.omega_max)

            # CBF pareado formal (bidirecional)
            if c > 1e-3:
                v_upper = min(v_upper, self.alpha_pair * h / c)
            elif c < -1e-3:
                v_lower = max(v_lower, self.alpha_pair * h / c)

            # Escape lateral quando o vizinho está “de lado” e perto
            if d < (R_pair + 0.8) and math.radians(60) < abs(phi) < math.radians(120):
                turn_dir = -1.0 if phi > 0.0 else 1.0
                w_des = sat(w_des + 0.5 * self.omega_max * turn_dir, -self.omega_max, self.omega_max)

        # Aplica limites combinados (v_lower <= v <= v_upper)
        v_cmd = sat(v_des, v_lower, v_upper)
        return v_cmd, w_des


    # ------------------------- Controle de movimento -------------------------
    def _compute_cmd(self):
        # 1. Sem objetivo, parar.
        if self.goal is None:
            return 0.0, 0.0

        # 2. Decide modo (desvio vs objetivo)
        obstacle_in_view = (self.scan is not None) and (self._min_range < self.safe_range_lidar)

        if obstacle_in_view:
            # --- MODO DESVIO: só campo repulsivo/tangencial ---
            ux_o, uy_o = self._repulsion_and_tangent_from_lidar()
            ux, uy = ux_o, uy_o
        else:
            # --- MODO OBJETIVO: só atrativo ---
            ux_g, uy_g, dist = self._attractive_to_goal()
            if dist == 0.0:
                return 0.0, 0.0
            ux, uy = ux_g, uy_g

        # 3. Normaliza vetor força -> v_max
        norm = math.hypot(ux, uy)
        if norm < 1e-6:
            return 0.0, 0.0
        ux *= self.v_max / norm
        uy *= self.v_max / norm

        # 4. Feedback linearization
        v =  math.cos(self.th) * ux + math.sin(self.th) * uy
        w = (-math.sin(self.th) / self.d) * ux + (math.cos(self.th) / self.d) * uy
        v = sat(v, -self.v_max, self.v_max)
        w = sat(w, -self.omega_max, self.omega_max)

        # 5. PROJEÇÃO CBF (LIDAR + pares, bidirecional)
        v, w = self._cbf_project_multi(v, w)

        # 6. Partida escalonada + rampa
        t = time.time() - self.t_start
        t_gate = self.self_idx * self.start_stagger_s
        if t < t_gate:
            v, w = 0.0, 0.0
        else:
            ramp = sat((t - t_gate) / max(1e-3, self.speed_ramp_s), 0.0, 1.0)
            v *= ramp
            w *= (0.7 + 0.3*ramp)

        # 7. Amortecimento global
        v *= (1.0 - self.k_damp)
        w *= (1.0 - 0.5 * self.k_damp)
        return v, w

    # ------------------------------- Loop ------------------------------------
    def spin(self):
        tw = Twist()
        t_last_log = time.time()
        while not rospy.is_shutdown():
            v_cmd, w_cmd = (0.0, 0.0)
            if self.have_pose:
                v_cmd, w_cmd = self._compute_cmd()

            self._battery_update(v_cmd, w_cmd)

            tw.linear.x  = v_cmd
            tw.angular.z = w_cmd
            self.pub_cmd_sim.publish(tw)
            self.pub_cmd_logical.publish(tw)

            now = time.time()
            if now - t_last_log > 1.0:
                if self.goal is not None:
                    dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
                    rospy.loginfo(
                        f"[{self.name}] dmin={self._min_range:.2f} "
                        f"pose=({self.x:.1f},{self.y:.1f},{self.th:.2f}) "
                        f"goal=({self.goal[0]:.1f},{self.goal[1]:.1f}) d={dist:.1f} "
                        f"cmd=({v_cmd:.2f},{w_cmd:.2f})"
                    )
                t_last_log = now

            if self.goal is not None:
                dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
                if dist < self.goal_eps:
                    rospy.loginfo(f"[{self.name}] Objetivo atingido.")
                    self.goal = None

            self.rate.sleep()

# -------------------------------- Execução -----------------------------------
def main():
    rospy.init_node("uav_agent")
    robot_name = rospy.get_param("~robot_name", None)
    if robot_name is None:
        ns = rospy.get_namespace().strip("/")
        robot_name = ns if ns else "vant_0"
    VANT(name=robot_name).spin()

if __name__ == "__main__":
    main()
