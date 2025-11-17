#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import re
import threading
import numpy as np

import rospy
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Header
from tf.transformations import euler_from_quaternion

from airspace_control.srv import (
    GetBattery, GetBatteryResponse,
    GotoXY, GotoXYRequest, GotoXYResponse
)

# ---------------------------------- Utils -----------------------------------
def sat(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

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
            if t in existing:
                return t
        time.sleep(poll_dt)
    existing = _published_topics_snapshot()
    for t in cands:
        if t in existing:
            return t
    return None

def _extract_index_from_name(name):
    m = re.search(r'_(\d+)$', str(name))
    return int(m.group(1)) if m else 0

# ---------------------------------- VANT ------------------------------------
class VANT:

    def __init__(self, name: str, ros_node=None):
        self.name = name
        self.ros_node = ros_node if ros_node is not None else rospy

        # Estado
        self.x = self.y = self.th = 0.0
        self.vx = self.vy = self.vth = 0.0
        self.have_pose = False
        self._pose_lock = threading.Lock()

        # LIDAR (max range 5.0m)
        self.scan = None
        self._scan_lock = threading.Lock()
        self._min_range = float("inf")
        self._min_bearing = 0.0

        # Objetivo e navegação
        self.goal = None
        self._goal_lock = threading.Lock()
        self.goal_eps = self.ros_node.get_param("~goal_tolerance", 1.0) # OK
        self._pending_release_event = None
        self._path = []
        self._current_path_index = 0

        # --- AJUSTES BASEADOS NO ROBÔ DE 1.2m E LIDAR DE 5.0m ---
        
        # Controladores (GANHOS MUITO MAIS SUAVES PARA ROBÔ GRANDE)
        self.Kp_linear = self.ros_node.get_param("~Kp_linear", 0.8)  # Reduzido de 1.5
        self.Kp_angular = self.ros_node.get_param("~Kp_angular", 1.8) # Reduzido de 2.5
        self.Ki_angular = self.ros_node.get_param("~Ki_angular", 0.01) # Quase zero para evitar overshoot
        self.Kd_angular = self.ros_node.get_param("~Kd_angular", 0.6)  # Aumentado para amortecer
        self._angular_integral = 0.0
        self._last_angular_error = 0.0
        
        # Limites dinâmicos (Assumindo um robô de estágio robusto)
        self.v_max = self.ros_node.get_param("~v_max", 2.0) # Reduzido de 2.5
        self.omega_max = self.ros_node.get_param("~omega_max", 1.5) # Reduzido de 1.8
        self.accel_max = self.ros_node.get_param("~accel_max", 0.8) # Reduzido
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0

        # Segurança CBF (Criticamente reajustado para o robô de 1.2m)
        self.robot_radius = self.ros_node.get_param("~robot_radius", 0.65) # Metade do tamanho do erratic (1.20) + margem
        self.emergency_stop_range = self.ros_node.get_param("~emergency_stop_range", 0.8) # Mantido em 0.8
        self.safety_margin = self.ros_node.get_param("~safety_margin", 1.2) # Inicia evasão assim que um objeto toca o raio do robô (0.65+0.55)
        self.slow_down_range = self.ros_node.get_param("~slow_down_range", 3.0) # Começa a desacelerar a 3.0m
        
        # Parâmetros de campo potencial melhorados
        self.k_att = self.ros_node.get_param("~k_att", 1.0) # Mais suave
        self.k_rep = self.ros_node.get_param("~k_rep", 0.8) # Reduzido MUITO para evitar ser "preso"
        self.repulsive_max = self.ros_node.get_param("~repulsive_max", 1.0)
        self.eta = self.ros_node.get_param("~eta", 0.8)

        # Navegação inteligente
        self.lookahead_dist = self.ros_node.get_param("~lookahead_dist", 4.0) # Distância de lookahead maior (robô maior)
        self.corridor_width = self.ros_node.get_param("~corridor_width", 3.0) # Corredor maior
        self.obstacle_influence_range = self.ros_node.get_param("~obstacle_influence_range", 4.5) # Obstáculos influenciam até o limite do LIDAR (5.0m)

        # Evasão de obstáculos
        self.escape_maneuver = False
        self.escape_direction = 0
        self.escape_start_time = 0
        self.escape_duration = self.ros_node.get_param("~escape_duration", 3.0) # Duração da manobra maior

        # Detecção de vizinhos (inalterado)
        self.neigh = {}
        self._neigh_lock = threading.Lock()
        self.communication_range = self.ros_node.get_param("~communication_range", 10.0)
        self.coordination_enabled = self.ros_node.get_param("~coordination_enabled", True)

        # Partida segura (inalterado)
        self.t_start = time.time()
        self.start_stagger_s = self.ros_node.get_param("~start_stagger_s", 2.0)
        self.speed_ramp_s = self.ros_node.get_param("~speed_ramp_s", 3.0)

        # Bateria (inalterado)
        self.voltage_nom = self.ros_node.get_param("~voltage_nom", 22.2)
        self.capacity_Wh = self.ros_node.get_param("~capacity_Wh", 180.0)
        self.soc = 1.0
        self.i_base = self.ros_node.get_param("~i_base", 1.8)
        self.i_vgain = self.ros_node.get_param("~i_vgain", 2.5)
        self.i_wgain = self.ros_node.get_param("~i_wgain", 1.8)
        self.last_batt_ts = time.time()
        self._last_current = 0.0

        # Autodetecção de tópicos
        idx = _extract_index_from_name(self.name)
        self.self_idx = idx
        ns_logical = f"/{self.name}"
        ns_robot = f"/robot_{idx}"
        
        # ... (Resto da configuração de Tópicos e Serviços inalterada) ...

        odom_candidates = [
            f"{ns_logical}/odom", f"{ns_logical}/base_pose_ground_truth",
            f"{ns_robot}/odom", f"{ns_robot}/base_pose_ground_truth",
        ]
        scan_candidates = [f"{ns_logical}/base_scan", f"{ns_robot}/base_scan"]
        cmd_candidates = [f"{ns_robot}/cmd_vel", f"{ns_logical}/cmd_vel"]

        odom_topic = _pick_first_existing(odom_candidates) or odom_candidates[-1]
        scan_topic = _pick_first_existing(scan_candidates) or scan_candidates[-1]
        cmd_sim_topic = cmd_candidates[0]

        # Publishers
        self.pub_cmd_logical = self.ros_node.Publisher(f"{ns_logical}/cmd_vel", Twist, queue_size=10)
        self.pub_cmd_sim = self.ros_node.Publisher(cmd_sim_topic, Twist, queue_size=10)
        self.pub_event_out = self.ros_node.Publisher("/event", String, queue_size=10)
        self.pub_path = self.ros_node.Publisher(f"{ns_logical}/planned_path", Path, queue_size=10)

        # Subscribers
        self.ros_node.Subscriber(odom_topic, Odometry, self._cb_odom, queue_size=10)
        self.ros_node.Subscriber(scan_topic, LaserScan, self._cb_scan, queue_size=10)

        # Esperar pela primeira pose
        self._wait_for_initial_pose(odom_topic)

        # Services
        self.srv_batt = self.ros_node.Service(f"{ns_logical}/get_battery", GetBattery, self._srv_get_battery)
        self.srv_goto = self.ros_node.Service(f"{ns_logical}/goto_xy", GotoXY, self._srv_goto_xy)

        # Descoberta de vizinhos
        self._setup_neighbors()

        self.rate = self.ros_node.Rate(self.ros_node.get_param("~rate_hz", 25.0))
        
        self.ros_node.loginfo(f"[{self.name}] ✅ Sistema inicializado - Pronto para operação")
        self.ros_node.loginfo(f"[{self.name}] 📍 Tópicos: odom='{odom_topic}', scan='{scan_topic}'")

    def _wait_for_initial_pose(self, odom_topic):
        t0 = time.time()
        while not self.have_pose and (time.time() - t0) < 8.0 and not self.ros_node.is_shutdown():
            self.ros_node.loginfo_throttle(2, f"[{self.name}] ⏳ Aguardando pose inicial...")
            self.ros_node.sleep(0.1)
        
        if not self.have_pose:
            self.ros_node.logwarn(f"[{self.name}] ⚠️  Pose inicial não recebida, continuando...")

    def _setup_neighbors(self):
        topics = _published_topics_snapshot()
        indices = set()
        
        for t in topics:
            for pattern in [r'/robot_(\d+)/odom', r'/vant_(\d+)/odom', r'/vant_(\d+)/base_pose_ground_truth']:
                m = re.search(pattern, t)
                if m: 
                    indices.add(int(m.group(1)))
        
        for k in sorted(indices):
            if k == self.self_idx: 
                continue
                
            cands = [f"/robot_{k}/odom", f"/vant_{k}/odom", f"/vant_{k}/base_pose_ground_truth"]
            topic_k = next((c for c in cands if c in topics), None)
            
            if topic_k:
                with self._neigh_lock:
                    self.neigh[k] = {"x": 0.0, "y": 0.0, "th": 0.0, "ts": 0.0, "has": False}
                self.ros_node.Subscriber(topic_k, Odometry, self._mk_cb_neigh(k), queue_size=5)
        
        if self.neigh:
            self.ros_node.loginfo(f"[{self.name}] 👥 Vizinhos detectados: {sorted(self.neigh.keys())}")

    def _mk_cb_neigh(self, k):
        def _cb(msg: Odometry):
            with self._neigh_lock:
                if k not in self.neigh:
                    self.neigh[k] = {"x": 0.0, "y": 0.0, "th": 0.0, "ts": 0.0, "has": False}
                
                p = self.neigh[k]
                p["x"] = msg.pose.pose.position.x
                p["y"] = msg.pose.pose.position.y
                
                q = msg.pose.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                p["th"] = yaw
                p["ts"] = time.time()
                p["has"] = True
        return _cb

    def _cb_odom(self, msg: Odometry):
        with self._pose_lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            
            q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.th = yaw
            
            self.vx = msg.twist.twist.linear.x
            self.vy = msg.twist.twist.linear.y
            self.vth = msg.twist.twist.angular.z
            
            self.have_pose = True

    def _cb_scan(self, msg: LaserScan):
        with self._scan_lock:
            self.scan = msg
            self._min_range = float("inf")
            self._min_bearing = 0.0
            
            if msg.ranges:
                angle = msg.angle_min
                for r in msg.ranges:
                    if r >= msg.range_min and r <= msg.range_max and r < self._min_range:
                        self._min_range = r
                        self._min_bearing = angle
                    angle += msg.angle_increment

    def _srv_get_battery(self, _req):
        self._battery_update(0.0, 0.0)
        rem_min = self._estimate_remaining_minutes()
        return GetBatteryResponse(
            soc=self.soc,
            voltage=self.voltage_nom,
            current=self._last_current,
            remaining_minutes=rem_min,
        )

    def _srv_goto_xy(self, req: GotoXYRequest):
        with self._goal_lock:
            self.goal = (float(req.x), float(req.y))
            self._pending_release_event = req.release_event
            self._path = []
            self._current_path_index = 0
            self.escape_maneuver = False
            
            # --- CORREÇÃO: Resetar o estado do controlador PID angular (ESSENCIAL) ---
            self._angular_integral = 0.0
            self._last_angular_error = 0.0
            # -----------------------------------------------------------------------
            
        self.ros_node.loginfo(f"[{self.name}] 🎯 Novo objetivo: ({req.x:.2f}, {req.y:.2f})")
        if req.release_event:
            self.ros_node.loginfo(f"[{self.name}] 📡 Evento de liberação: '{req.release_event}'")
        
        return GotoXYResponse(accepted=True, message="Objetivo recebido e planejamento iniciado")
    
    def _battery_update(self, v_cmd, w_cmd):
        now = time.time()
        dt = max(0.001, now - self.last_batt_ts)
        self.last_batt_ts = now
        
        power = (self.i_base + self.i_vgain * abs(v_cmd) + self.i_wgain * abs(w_cmd)) * self.voltage_nom
        energy_used = power * (dt / 3600.0)
        self.soc = max(0.0, self.soc - (energy_used / self.capacity_Wh))
        self._last_current = power / self.voltage_nom if self.voltage_nom > 0 else 0.0

    def _estimate_remaining_minutes(self):
        avg_power = (self.i_base + self.i_vgain * 0.6 * self.v_max + self.i_wgain * 0.4 * self.omega_max) * self.voltage_nom
        if avg_power <= 1e-6:
            return float("inf")
        remaining_energy = self.soc * self.capacity_Wh
        return max(0.0, 60.0 * (remaining_energy / avg_power))

    def _compute_potential_field(self):
        if self.goal is None or not self.have_pose:
            return 0.0, 0.0

        goal_x, goal_y = self.goal
        dx_goal = goal_x - self.x
        dy_goal = goal_y - self.y
        dist_to_goal = math.hypot(dx_goal, dy_goal)
        
        if dist_to_goal < 1e-6:
            return 0.0, 0.0

        # Campo atrativo suavizado
        attractive_gain = min(1.0, dist_to_goal / 5.0)
        f_att_x = self.k_att * attractive_gain * (dx_goal / dist_to_goal)
        f_att_y = self.k_att * attractive_gain * (dy_goal / dist_to_goal)

        # Campo repulsivo do LIDAR
        f_rep_x, f_rep_y = 0.0, 0.0
        
        if self.scan and len(self.scan.ranges) > 0:
            angle = self.scan.angle_min
            for r in self.scan.ranges:
                if r < self.scan.range_min or r > self.scan.range_max or not math.isfinite(r):
                    angle += self.scan.angle_increment
                    continue

                if r < self.obstacle_influence_range:
                    # Converter para coordenadas globais
                    obstacle_angle = self.th + angle
                    obstacle_x = self.x + r * math.cos(obstacle_angle)
                    obstacle_y = self.y + r * math.sin(obstacle_angle)
                    
                    dx_obs = self.x - obstacle_x
                    dy_obs = self.y - obstacle_y
                    dist_to_obs = math.hypot(dx_obs, dy_obs)
                    
                    if dist_to_obs < 1e-6:
                        angle += self.scan.angle_increment
                        continue

                    # Força repulsiva inversamente proporcional à distância
                    repulsive_strength = min(self.repulsive_max, 
                                           self.k_rep * (1.0 / dist_to_obs - 1.0 / self.obstacle_influence_range) / (dist_to_obs ** 2))
                    
                    if dist_to_obs < self.emergency_stop_range:
                        repulsive_strength *= 3.0
                    elif dist_to_obs < self.slow_down_range:
                        repulsive_strength *= 1.5

                    f_rep_x += repulsive_strength * (dx_obs / dist_to_obs)
                    f_rep_y += repulsive_strength * (dy_obs / dist_to_obs)

                angle += self.scan.angle_increment

        # Campo repulsivo de vizinhos
        with self._neigh_lock:
            for k, p in self.neigh.items():
                if not p["has"] or (time.time() - p["ts"]) > 2.0:
                    continue

                dx_neigh = self.x - p["x"]
                dy_neigh = self.y - p["y"]
                dist_to_neigh = math.hypot(dx_neigh, dy_neigh)

                if dist_to_neigh < self.communication_range and dist_to_neigh > 1e-6:
                    neighbor_repulsion = min(self.repulsive_max * 0.7,
                                           self.k_rep * 0.5 * (1.0 / dist_to_neigh) / (dist_to_neigh ** 2))
                    
                    f_rep_x += neighbor_repulsion * (dx_neigh / dist_to_neigh)
                    f_rep_y += neighbor_repulsion * (dy_neigh / dist_to_neigh)

        # Combinação dos campos
        total_fx = f_att_x + f_rep_x
        total_fy = f_att_y + f_rep_y

        return total_fx, total_fy

    def _check_emergency_stop(self):
        if not self.scan or len(self.scan.ranges) == 0:
            return False

        # Verificar se há obstáculos muito próximos
        min_safe_distance = self.emergency_stop_range
        for r in self.scan.ranges:
            if r < min_safe_distance and r >= self.scan.range_min:
                return True

        return False

    def _smooth_control_command(self, v_desired, w_desired, dt):
        # Limitação de aceleração
        max_dv = self.accel_max * dt
        max_dw = self.omega_max * 0.8 * dt
        
        v_smooth = self._last_v_cmd + sat(v_desired - self._last_v_cmd, -max_dv, max_dv)
        w_smooth = self._last_w_cmd + sat(w_desired - self._last_w_cmd, -max_dw, max_dw)
        
        self._last_v_cmd = v_smooth
        self._last_w_cmd = w_smooth
        
        return v_smooth, w_smooth

    def _compute_control_command(self):
        """
        Calcula os comandos de velocidade linear (v_cmd) e angular (w_cmd)
        usando a navegação por Campos Potenciais Artificiais (APF)
        combinada com controle PID angular e lógica de evasão/segurança.
        """
        if self.goal is None or not self.have_pose:
            return 0.0, 0.0

        # 1. Verificação de parada de emergência
        if self._check_emergency_stop():
            self.ros_node.logwarn_throttle(1, f"[{self.name}] 🚨 PARADA DE EMERGÊNCIA - Obstáculo muito próximo!")
            return 0.0, 0.0

        # 2. Calcular campo potencial (Força Resultante)
        fx, fy = self._compute_potential_field()
        
        # 3. Converter força em velocidade desejada
        desired_angle = math.atan2(fy, fx)
        desired_speed = min(self.v_max, math.hypot(fx, fy) * 0.5)
        
        # 4. Controle PID Angular
        angle_error = wrap_pi(desired_angle - self.th)
        
        # Termos do PID
        self._angular_integral += angle_error
        # Saturação do integral apenas (Ki é muito pequeno)
        self._angular_integral = sat(self._angular_integral, -2.0, 2.0) 
        
        angular_derivative = angle_error - self._last_angular_error 
        
        w_cmd = (self.Kp_angular * angle_error + 
                 self.Ki_angular * self._angular_integral + 
                 self.Kd_angular * angular_derivative)
        
        self._last_angular_error = angle_error
        
        # 5. Controle Linear (v_cmd) e Alinhamento Suavizado (CORREÇÃO CHAVE)
        alignment = math.cos(angle_error)
        
        # Garante um fator mínimo de velocidade linear (0.2)
        # para evitar que o robô zere 'v_cmd' e trave girando no lugar.
        speed_factor = max(0.2, alignment)
        speed_factor = min(1.0, speed_factor)
        
        v_cmd = desired_speed * speed_factor
        
        # 6. Redução de Velocidade Próximo a Obstáculos
        if self._min_range < self.slow_down_range:
            obstacle_factor = min(1.0, self._min_range / self.slow_down_range)
            v_cmd *= obstacle_factor
            # Aumenta a agressividade na rotação para escapar
            w_cmd *= (1.0 + (1.0 - obstacle_factor)) 

        # 7. Manobra de Evasão (Se necessário)
        if self._min_range < self.safety_margin and not self.escape_maneuver:
            self.escape_maneuver = True
            self.escape_start_time = time.time()
            self.escape_direction = 1.0 if self._min_bearing > 0 else -1.0
            self.ros_node.loginfo(f"[{self.name}] 🌀 Iniciando manobra de evasão")
        
        if self.escape_maneuver:
            if time.time() - self.escape_start_time < self.escape_duration:
                v_cmd *= 0.3  # Reduzir velocidade durante a manobra
                w_cmd = self.escape_direction * self.omega_max * 0.8 # Gira forte para escapar
            else:
                self.escape_maneuver = False
                self.ros_node.loginfo(f"[{self.name}] ✅ Manobra de evasão concluída")

        # 8. Saturação final
        v_cmd = sat(v_cmd, 0.0, self.v_max)
        w_cmd = sat(w_cmd, -self.omega_max, self.omega_max)

        # 9. Partida escalonada (Staggered Start)
        t_since_start = time.time() - self.t_start
        if t_since_start < self.start_stagger_s * (self.self_idx + 1):
            v_cmd, w_cmd = 0.0, 0.0
        elif t_since_start < self.start_stagger_s * (self.self_idx + 1) + self.speed_ramp_s:
            ramp_factor = (t_since_start - self.start_stagger_s * (self.self_idx + 1)) / self.speed_ramp_s
            v_cmd *= ramp_factor
            w_cmd *= ramp_factor

        return v_cmd, w_cmd

    def _check_goal_reached(self):
        if self.goal is None or not self.have_pose:
            return False

        goal_x, goal_y = self.goal
        distance = math.hypot(goal_x - self.x, goal_y - self.y)
        
        # Critérios para considerar objetivo atingido
        position_reached = distance < self.goal_eps
        low_speed = math.hypot(self.vx, self.vy) < 0.2
        
        return position_reached and low_speed

    def _publish_path(self):
        if not self.goal or not self._path:
            return
            
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        
        for point in self._path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position = Point(x=point[0], y=point[1], z=0)
            path_msg.poses.append(pose)
            
        self.pub_path.publish(path_msg)

    def _stop_movement(self):
        """Para o movimento do VANT imediatamente"""
        self.ros_node.loginfo(f"[{self.name}] 🛑 Parada de movimento solicitada")
        
        # Publicar comando zero
        twist = Twist()
        self.pub_cmd_sim.publish(twist)
        self.pub_cmd_logical.publish(twist)
        
        # Resetar estado de controle
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0
        self._angular_integral = 0.0
        self._last_angular_error = 0.0
        self.escape_maneuver = False

    def spin(self):
        """Loop principal de controle"""
        last_control_time = time.time()
        last_log_time = 0
        goal_reached_logged = False

        self.ros_node.loginfo(f"[{self.name}] 🚀 Iniciando loop de controle principal")

        while not self.ros_node.is_shutdown():
            current_time = time.time()
            dt = max(0.001, current_time - last_control_time)
            last_control_time = current_time

            # Computar comando de controle
            v_cmd, w_cmd = self._compute_control_command()

            # Verificar se objetivo foi atingido
            if self._check_goal_reached():
                if not goal_reached_logged:
                    self.ros_node.loginfo(f"[{self.name}] ✅ OBJETIVO ATINGIDO!")
                    goal_reached_logged = True
                    
                    # Publicar evento de liberação se existir
                    if self._pending_release_event:
                        event_msg = String(data=self._pending_release_event)
                        self.pub_event_out.publish(event_msg)
                        self.ros_node.loginfo(f"[{self.name}] 📡 Evento publicado: '{self._pending_release_event}'")
                        self._pending_release_event = None
                    
                    # Parar movimento
                    v_cmd, w_cmd = 0.0, 0.0
                    with self._goal_lock:
                        self.goal = None
                    break
            else:
                goal_reached_logged = False

            # Suavizar comando
            v_cmd, w_cmd = self._smooth_control_command(v_cmd, w_cmd, dt)

            # Atualizar bateria
            self._battery_update(v_cmd, w_cmd)

            # Publicar comando
            twist = Twist()
            twist.linear.x = v_cmd
            twist.angular.z = w_cmd
            self.pub_cmd_sim.publish(twist)
            self.pub_cmd_logical.publish(twist)

            # Log periódico
            if current_time - last_log_time > 2.0:  # Log a cada 2 segundos
                if self.goal:
                    goal_x, goal_y = self.goal
                    distance = math.hypot(goal_x - self.x, goal_y - self.y)
                    self.ros_node.loginfo(
                        f"[{self.name}] 📊 Estado: pos=({self.x:.1f}, {self.y:.1f}) "
                        f"goal=({goal_x:.1f}, {goal_y:.1f}) dist={distance:.1f}m "
                        f"cmd=({v_cmd:.2f}, {w_cmd:.2f}) soc={self.soc:.1%}"
                    )
                last_log_time = current_time

            self.rate.sleep()

        self.ros_node.loginfo(f"[{self.name}] 🔚 Loop de controle encerrado")

def main():
    rospy.init_node("uav_agent")
    robot_name = rospy.get_param("~robot_name", None)
    if robot_name is None:
        ns = rospy.get_namespace().strip("/")
        robot_name = ns if ns else "vant_0"

    try:
        vant = VANT(name=robot_name, ros_node=rospy)
        vant.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"[{robot_name}] ❌ Erro fatal: {e}")
        raise

if __name__ == "__main__":
    main()