#!/usr/bin/env python3
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.insert(0, parent_path)

import rospy
from std_msgs.msg import String
from tkinter import *
from tkinter import ttk
import rosnode
import time

class AutomatonInfo:
    def __init__(self, name):
        self.name = name
        self.current_state = None
        self.previous_state = None
        self.entry_time = None
        self.possible_events = []
        self.is_marked = False
        self.type = "supervisor" if "supervisor" in name.lower() else "plant"
        self.label_state = None
        self.label_time = None
        self.dropdown = None

class ControlPanel:
    def __init__(self):
        rospy.init_node("control_panel", anonymous=False)
        self.automata = {}
        self.publishers = {}
        self.dropdown_vars = {}
        self.frames = {}
        self.limit = 0

        self.root = Tk()
        self.root.title("Control Panel - UltraDES + ROS")
        self.root.geometry("800x600")

        self.canvas = Canvas(self.root)
        self.scrollbar = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Top buttons
        top_frame = Frame(self.scrollable_frame)
        top_frame.pack(pady=5, fill="x")

        Button(top_frame, text="Refresh Automata", command=self.refresh_automata).pack(side="left", padx=10)
        Button(top_frame, text="Send PING to All", command=self.ping_all).pack(side="left", padx=10)

        self.refresh_automata()
        self.update_interface()
        self.root.mainloop()

    def refresh_automata(self):
        try:
            all_nodes = rosnode.get_node_names()
            valid_nodes = [
                n.strip("/") for n in all_nodes
                if n not in ["/rosout", "/control_panel"]
            ]
            self.limit = len(valid_nodes)

            for name in valid_nodes:
                if name not in self.automata:
                    self.automata[name] = AutomatonInfo(name)
                    rospy.Subscriber(f"/{name}/state", String, self.callback_state, callback_args=name)
                    rospy.Subscriber(f"/{name}/possible_events", String, self.callback_events, callback_args=name)
                    rospy.Subscriber(f"/{name}/is_marked", String, self.callback_marked, callback_args=name)
                    rospy.loginfo(f"[Panel] New automaton detected: {name}")

                    self.publishers[name] = rospy.Publisher("/event", String, queue_size=10)
                    rospy.sleep(0.2)
                    self.publishers[name].publish("ping")
        except Exception as e:
            rospy.logwarn(f"[Panel] Error while refreshing automata: {e}")

    def ping_all(self):
        for name, pub in self.publishers.items():
            pub.publish("ping")
        rospy.loginfo("[Panel] Ping sent to all automata.")

    def callback_state(self, msg, name):
        def update():
            try:
                info = self.automata[name]
                new_state = msg.data
                if new_state != info.current_state:
                    info.previous_state = info.current_state
                    info.current_state = new_state
                    info.entry_time = rospy.get_time()
                self.update_individual_interface(name)
            except Exception as e:
                rospy.logwarn(f"[Panel] Error in state callback for '{name}': {e}")
        self.root.after(0, update)

    def callback_events(self, msg, name):
        def update():
            try:
                events = msg.data.strip().split(",") if msg.data else []
                self.automata[name].possible_events = events
                self.update_individual_interface(name)
            except Exception as e:
                rospy.logwarn(f"[Panel] Error in events callback for '{name}': {e}")
        self.root.after(0, update)

    def callback_marked(self, msg, name):
        def update():
            try:
                self.automata[name].is_marked = msg.data == "True"
                self.update_individual_interface(name)
            except Exception as e:
                rospy.logwarn(f"[Panel] Error in marked callback for '{name}': {e}")
        self.root.after(0, update)

    def send_event(self, name):
        event = self.dropdown_vars[name].get()
        if not event:
            rospy.logwarn(f"[Panel] No event selected for '{name}'")
            return
        if name not in self.publishers:
            self.publishers[name] = rospy.Publisher("/event", String, queue_size=10)
            rospy.sleep(0.5)
        self.publishers[name].publish(event)
        rospy.loginfo(f"[Panel] Sent event '{event}' to '{name}'")

    def update_interface(self):
        valid_names = [
            name for name, info in self.automata.items()
            if info.current_state not in [None, "???"]
        ]

        sorted_names = sorted(valid_names, key=lambda n: (0 if "supervisor" in n.lower() else 1, n.lower()))
        limited_names = sorted_names[:self.limit]

        for name in limited_names:
            self.update_individual_interface(name)

        for name in limited_names:
            if name in self.frames:
                self.frames[name].pack_forget()
                self.frames[name].pack(padx=10, pady=5, fill="x")

        self.root.after(1000, self.update_interface)

    def update_individual_interface(self, name):
        info = self.automata[name]

        if info.current_state in [None, "???"]:
            return

        has_supervisor = any(a.type == "supervisor" for a in self.automata.values())

        if name not in self.frames:
            frame = LabelFrame(self.scrollable_frame, padx=10, pady=10)
            frame.pack(padx=10, pady=5, fill="x")
            self.frames[name] = frame

            label_name = Label(frame, text=f"{name.upper()} — State:", font=("Arial", 12, "bold"))
            label_name.pack(anchor="w")
            info.label_state = Label(frame, text="???", font=("Arial", 12))
            info.label_state.pack(anchor="w")

            info.label_time = Label(frame, text="Time in state: 0s", font=("Arial", 10, "italic"))
            info.label_time.pack(anchor="w")

            inner_frame = Frame(frame)
            inner_frame.pack(anchor="w", pady=5)
            info.inner_frame = inner_frame

            self.dropdown_vars[name] = StringVar()
            info.dropdown = ttk.Combobox(inner_frame, textvariable=self.dropdown_vars[name], state="readonly", width=40)
            info.dropdown.pack(side="left", padx=5)

            info.send_button = Button(inner_frame, text="SEND", command=lambda n=name: self.send_event(n))
            info.send_button.pack(side="left", padx=5)

        frame = self.frames[name]
        color = "#FFFF99" if info.type == "supervisor" and not info.is_marked else \
                "#90EE90" if info.is_marked else "#D3D3D3"
        frame.config(bg=color)
        info.label_state.config(text=info.current_state, bg=color)

        if info.entry_time:
            elapsed = int(rospy.get_time() - info.entry_time)
            info.label_time.config(text=f"Time in state: {elapsed}s", bg=color)

        if has_supervisor and info.type != "supervisor":
            info.inner_frame.pack_forget()
        else:
            events = info.possible_events
            if list(info.dropdown["values"]) != events:
                info.dropdown["values"] = events
                current_val = self.dropdown_vars[name].get()
                if current_val not in events and events:
                    self.dropdown_vars[name].set(events[0])
            info.inner_frame.pack(anchor="w", pady=5)

if __name__ == "__main__":
    ControlPanel()
