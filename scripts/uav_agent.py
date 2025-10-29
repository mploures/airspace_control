#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import argparse
from collections import defaultdict

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from airspace_control.srv import (
    GetBattery, GetBatteryRequest, GetBatteryResponse,
    GotoXY, GotoXYRequest, GotoXYResponse
)

def sat(value, vmin, vmax):
    return max(vmin, min(vmax, value))


def wrap_pi(a):
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a


class VANT:
    """
    Classe do VANT com duas interfaces ROS:
      - /<name>/get_battery (service GetBattery)
      - /<name>/goto_xy    (service GotoXY)
    Tópicos esperados do Stage (namespaced pelo nome do modelo):
      - /<name>/base_pose_ground_truth (nav_msgs/Odometry)
      - /<name>/base_scan              (sensor_msgs/LaserScan)
      - /<name>/cmd_vel                (geometry_msgs/Twist)
    """

    def __init__(self, name, all_agents):
        self.name = name
        self.all_agents = [a for a in all_agents if a != name]

        # Estado próprio
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.have_pose = False

        # LIDAR
        self.scan = None

        # Objetivo
        self.goal = None    # (x, y)
        self.goal_eps = rospy.get_param("~goal_tolerance", 0.8)  # tolerância de chegada (m)

        # Controle (feedback linearization)
        self.Kp = rospy.get_param("~Kp", 1.2)
        self.d  = rospy.get_param("~d", 0.80)  # distância virtual para FL

        # Limites de velocidade (em Stage as escalas podem ser grandes)
        self.v_max = rospy.get_param("~v_max", 3.0)      # m/s
        self.omega_max = rospy.get_param("~omega_max", 1.5)  # rad/s

        # Evitação de colisão
        self.safe_range_lidar = rospy.get_param("~safe_range_lidar", 2.5)  # m
        self.safe_range_agents = rospy.get_param("~safe_range_agents", 3.5)
        self.k_rep_lidar = rospy.get_param("~k_rep_lidar", 2.0)
        self.k_rep_agents = rospy.get_param("~k_rep_agents", 3.0)
        self.k_att = rospy.get_param("~k_att", 1.5)

        # Bateria (modelo simples)
        self.voltage_nom = rospy.get_param("~voltage_nom", 22.2)      # 6S LiPo ~ 22.2 V
        self.capacity_Wh = rospy.get_param("~capacity_Wh", 150.0)     # 150 Wh (exemplo)
        self.soc = 1.0                                                # 100%
        self.i_base = rospy.get_param("~i_base", 2.0)                 # Corrente base (A)
        self.i_vgain = rospy.get_param("~i_vgain", 3.0)               # A por (m/s)
        self.i_wgain = rospy.get_param("~i_wgain", 2.0)               # A por (rad/s)
        self.last_batt_ts = time.time()

        # Estados dos outros VANTs
        self.others = defaultdict(lambda: {"x": None, "y": None, "th": None, "t": 0.0})
        self._make_other_subs()

        # ROS I/O
        ns = f"/{self.name}"
        self.pub_cmd = rospy.Publisher(f"{ns}/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber(f"{ns}/base_pose_ground_truth", Odometry, self._cb_odom, queue_size=10)
        rospy.Subscriber(f"{ns}/base_scan", LaserScan, self._cb_scan, queue_size=10)

        # Services
        self.srv_batt = rospy.Service(f"{ns}/get_battery", GetBattery, self._srv_get_battery)
        self.srv_goto = rospy.Service(f"{ns}/goto_xy", GotoXY, self._srv_goto_xy)

        # Loop de controle
        self.rate = rospy.Rate(rospy.get_param("~rate_hz", 30.0))

        rospy.loginfo(f"[{self.name}] pronto. Outros agentes: {self.all_agents}")

    # ---------- Callbacks ----------
    def _cb_odom(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        # yaw z
        siny_cosp = 2*(qw*qz + qx*qy)
        cosy_cosp = 1 - 2*(qy*qy + qz*qz)
        self.th = math.atan2(siny_cosp, cosy_cosp)
        self.have_pose = True

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    def _make_other_subs(self):
        for other in self.all_agents:
            rospy.Subscriber(f"/{other}/base_pose_ground_truth", Odometry,
                             lambda m, name=other: self._cb_other_odom(name, m),
                             queue_size=5)

    def _cb_other_odom(self, other_name, msg: Odometry):
        self.others[other_name]["x"] = msg.pose.pose.position.x
        self.others[other_name]["y"] = msg.pose.pose.position.y
        self.others[other_name]["t"] = rospy.get_time()

    # ---------- Services ----------
    def _srv_get_battery(self, _req):
        self._battery_update(0.0, 0.0)  # atualiza por tempo decorrido mesmo sem comando
        rem_min = self._estimate_remaining_minutes()
        return GetBatteryResponse(
            soc=self.soc,
            voltage=self.voltage_nom,
            current=self._last_current if hasattr(self, "_last_current") else 0.0,
            remaining_minutes=rem_min
        )

    def _srv_goto_xy(self, req: GotoXYRequest):
        self.goal = (float(req.x), float(req.y))
        msg = f"Novo objetivo: ({self.goal[0]:.2f}, {self.goal[1]:.2f})"
        rospy.loginfo(f"[{self.name}] {msg}")
        return GotoXYResponse(accepted=True, message=msg)

    # ---------- Bateria ----------
    def _battery_update(self, v_cmd, w_cmd):
        """Atualiza SoC usando um modelo simples de corrente."""
        now = time.time()
        dt = max(1e-3, now - self.last_batt_ts)
        self.last_batt_ts = now

        # Corrente estimada (A)
        i = self.i_base + self.i_vgain*abs(v_cmd) + self.i_wgain*abs(w_cmd)
        self._last_current = i

        # Energia (Wh) consumida no intervalo dt
        wh = (i * self.voltage_nom) * (dt/3600.0)
        d_soc = wh / self.capacity_Wh
        self.soc = float(max(0.0, self.soc - d_soc))

    def _estimate_remaining_minutes(self):
        # Corrente média: assume v_cmd≈0.5*v_max e w_cmd≈0.5*omega_max
        i_est = self.i_base + self.i_vgain*(0.5*self.v_max) + self.i_wgain*(0.5*self.omega_max)
        p_est = i_est * self.voltage_nom            # W
        e_rem_Wh = self.soc * self.capacity_Wh      # Wh
        if p_est <= 1e-6:
            return float('inf')
        minutes = 60.0 * (e_rem_Wh / p_est)
        return max(0.0, minutes)

    # ---------- Planejamento/Controle ----------
    def _repulsion_from_lidar(self):
        if self.scan is None:
            return 0.0, 0.0
        u_rx, u_ry = 0.0, 0.0
        a = self.scan.angle_min
        for r in self.scan.ranges:
            if 0.05 < r < self.safe_range_lidar:
                # Direção da leitura no frame do robô
                dx = math.cos(self.th + a)
                dy = math.sin(self.th + a)
                gain = self.k_rep_lidar * (1.0 / max(0.1, r) - 1.0 / self.safe_range_lidar)
                gain = max(0.0, gain)
                # Repulsão é oposta à direção do obstáculo
                u_rx -= gain * dx
                u_ry -= gain * dy
            a += self.scan.angle_increment
        return u_rx, u_ry

    def _repulsion_from_agents(self):
        u_ax, u_ay = 0.0, 0.0
        for name, st in self.others.items():
            ox, oy = st["x"], st["y"]
            if ox is None:
                continue
            rx = self.x - ox
            ry = self.y - oy
            d = math.hypot(rx, ry)
            if d < 1e-3:
                continue
            if d < self.safe_range_agents:
                gain = self.k_rep_agents * (1.0 / d - 1.0 / self.safe_range_agents)
                gain = max(0.0, gain)
                u_ax += gain * (rx / d)
                u_ay += gain * (ry / d)
        return u_ax, u_ay

    def _attractive_to_goal(self):
        if self.goal is None:
            return 0.0, 0.0, 0.0
        gx, gy = self.goal
        ex = gx - self.x
        ey = gy - self.y
        dist = math.hypot(ex, ey)
        if dist < self.goal_eps:
            return 0.0, 0.0, dist
        # Vetor atrativo
        ux = self.k_att * ex
        uy = self.k_att * ey
        return ux, uy, dist

    def _compute_cmd(self):
        # Campo atrativo
        ux_g, uy_g, dist = self._attractive_to_goal()

        # Campos repulsivos
        ux_l, uy_l = self._repulsion_from_lidar()
        ux_a, uy_a = self._repulsion_from_agents()

        # Composição
        ux = ux_g + ux_l + ux_a
        uy = uy_g + uy_l + uy_a

        # Se não há objetivo, pare
        if self.goal is None or dist == 0.0:
            return 0.0, 0.0

        # Saturação em norma (velocidade plana)
        norm = math.hypot(ux, uy)
        if norm > 1e-6:
            ux = ux * (self.v_max / norm)
            uy = uy * (self.v_max / norm)

        # Feedback linearization → v, ω
        v = math.cos(self.th) * ux + math.sin(self.th) * uy
        w = (-math.sin(self.th) / self.d) * ux + (math.cos(self.th) / self.d) * uy

        v = sat(v, -self.v_max, self.v_max)
        w = sat(w, -self.omega_max, self.omega_max)
        return v, w

    def spin(self):
        tw = Twist()
        while not rospy.is_shutdown():
            v_cmd, w_cmd = 0.0, 0.0
            if self.have_pose:
                v_cmd, w_cmd = self._compute_cmd()

            # Atualiza bateria
            self._battery_update(v_cmd, w_cmd)

            # Publica comando
            tw.linear.x = v_cmd
            tw.angular.z = w_cmd
            self.pub_cmd.publish(tw)

            # Finaliza objetivo se chegou
            if self.goal is not None:
                dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
                if dist < self.goal_eps:
                    rospy.loginfo(f"[{self.name}] Objetivo atingido.")
                    self.goal = None

            self.rate.sleep()


def main():
    rospy.init_node("uav_agent")

    # Lê lista global de agentes (setada no script de world)
    agent_names = rospy.get_param("/airspace_control/agent_names", [])
    # Nome do VANT (parâmetro privado)
    robot_name = rospy.get_param("~robot_name", None)
    if robot_name is None:
        # Tentativa: inferir por namespace do nó
        ns = rospy.get_namespace().strip("/")
        robot_name = ns if ns else "vant_0"

    # Se não houver parâmetro global, tente deduzir pelos padrões
    if not agent_names:
        # fallback para 5 agentes
        agent_names = [f"vant_{i}" for i in range(5)]

    agent = VANT(name=robot_name, all_agents=agent_names)
    agent.spin()


if __name__ == "__main__":
    main()
