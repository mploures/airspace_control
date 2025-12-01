#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import rospy
from std_msgs.msg import String

from tkinter import *
from tkinter import ttk, messagebox

GRAFO_PATH = "/home/mploures/catkin_ws/src/airspace_control/graph/sistema_logistico/grafo_recortado.txt"


def carregar_fornecedores_clientes(grafo_path: str):
    """
    Lê o arquivo de grafo e retorna duas listas:
      - fornecedores: ['FORNECEDOR_0', 'FORNECEDOR_1', ...]
      - clientes:     ['CLIENTE_0', 'CLIENTE_1', ...]
    """
    fornecedores = []
    clientes = []

    if not os.path.isfile(grafo_path):
        raise FileNotFoundError(f"Arquivo de grafo não encontrado: {grafo_path}")

    with open(grafo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("tipo"):
                # Cabeçalho: tipo,label,(x,y),conexoes
                continue

            # Só precisamos dos dois primeiros campos: tipo, label
            parts = line.split(",")
            if len(parts) < 2:
                continue

            tipo = parts[0].strip().upper()
            label = parts[1].strip()

            if tipo == "FORNECEDOR":
                fornecedores.append(label)
            elif tipo == "CLIENTE":
                clientes.append(label)

    # Ordena só pra ficar bonitinho no combo
    fornecedores = sorted(set(fornecedores))
    clientes = sorted(set(clientes))

    return fornecedores, clientes


class TaskPanel:
    def __init__(self, master):
        self.master = master
        self.master.title("Task Panel - Tarefas para UTM")
        self.master.geometry("500x260")

        # -------- ROS --------
        rospy.init_node("task_panel", anonymous=False)
        self.pub_tarefas = rospy.Publisher("/task_todo", String, queue_size=10)

        # -------- Dados do grafo --------
        try:
            self.fornecedores, self.clientes = carregar_fornecedores_clientes(GRAFO_PATH)
        except Exception as e:
            self.fornecedores, self.clientes = [], []
            rospy.logerr(f"[task_panel] Erro ao carregar grafo: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar grafo:\n{e}")

        # -------- UI --------
        self._build_ui()

        # Atualiza status ROS periodicamente (só pra detectar shutdown)
        self._update_ros()

    def _build_ui(self):
        frame = Frame(self.master, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        # Info do grafo
        Label(
            frame,
            text=f"Grafo: {os.path.basename(GRAFO_PATH)}",
            font=("Arial", 10, "italic")
        ).pack(anchor="w")

        Label(
            frame,
            text=f"Fornecedores: {len(self.fornecedores)} | Clientes: {len(self.clientes)}",
            font=("Arial", 10)
        ).pack(anchor="w", pady=(0, 10))

        # Linha fornecedor
        row1 = Frame(frame)
        row1.pack(fill="x", pady=5)

        Label(row1, text="Fornecedor:", font=("Arial", 11)).pack(side="left")
        self.var_fornecedor = StringVar()
        self.combo_fornecedor = ttk.Combobox(
            row1,
            textvariable=self.var_fornecedor,
            state="readonly",
            width=25,
            values=self.fornecedores
        )
        self.combo_fornecedor.pack(side="left", padx=8)

        # Linha cliente
        row2 = Frame(frame)
        row2.pack(fill="x", pady=5)

        Label(row2, text="Cliente:", font=("Arial", 11)).pack(side="left")
        self.var_cliente = StringVar()
        self.combo_cliente = ttk.Combobox(
            row2,
            textvariable=self.var_cliente,
            state="readonly",
            width=25,
            values=self.clientes
        )
        self.combo_cliente.pack(side="left", padx=26)

        # Botão enviar
        row3 = Frame(frame)
        row3.pack(fill="x", pady=15)

        self.btn_enviar = Button(
            row3,
            text="Enviar tarefa",
            font=("Arial", 11, "bold"),
            command=self._enviar_tarefa
        )
        self.btn_enviar.pack(side="left", padx=5)

        # Status
        self.label_status = Label(
            frame,
            text="Status: aguardando seleção...",
            font=("Arial", 10),
            fg="blue"
        )
        self.label_status.pack(anchor="w", pady=5)

        # Fecha limpo
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        # Se não carregou nada, desabilita botão
        if not self.fornecedores or not self.clientes:
            self.btn_enviar.config(state="disabled")
            self.label_status.config(
                text="Status: ERRO ao carregar fornecedores/clientes. Veja o terminal.",
                fg="red"
            )

    def _enviar_tarefa(self):
        fornecedor = self.var_fornecedor.get().strip()
        cliente = self.var_cliente.get().strip()

        if not fornecedor:
            messagebox.showwarning("Atenção", "Selecione um FORNECEDOR.")
            return
        if not cliente:
            messagebox.showwarning("Atenção", "Selecione um CLIENTE.")
            return

        payload = f"{fornecedor},{cliente}"
        try:
            msg = String(data=payload)
            self.pub_tarefas.publish(msg)
            rospy.loginfo(f"[task_panel] Tarefa enviada em /task_todo: '{payload}'")
            self.label_status.config(
                text=f"Status: tarefa enviada -> {payload}",
                fg="green"
            )
        except Exception as e:
            rospy.logerr(f"[task_panel] Erro ao publicar tarefa: {e}")
            self.label_status.config(
                text=f"Status: erro ao publicar tarefa (veja terminal).",
                fg="red"
            )

    def _update_ros(self):
        """
        Checa se o ROS foi encerrado e, se sim, fecha o painel.
        """
        if rospy.is_shutdown():
            try:
                self.master.destroy()
            except Exception:
                pass
            return

        # Agenda próxima checagem
        self.master.after(500, self._update_ros)

    def _on_close(self):
        try:
            rospy.signal_shutdown("Janela fechada pelo usuário")
        except Exception:
            pass
        try:
            self.master.destroy()
        except Exception:
            pass


def main():
    root = Tk()
    app = TaskPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
