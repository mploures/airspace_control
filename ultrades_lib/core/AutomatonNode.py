#!/usr/bin/env python3
import os
import sys
from datetime import datetime

# Dynamic path setup for UltraDES modules and local utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rospy
from std_msgs.msg import String
import ultrades
from ultrades.automata import *

from .utils import load_from_json 

class AutomatonNode:
    def __init__(self, name, automaton_path=None, dfa_obj=None, role="plant"):
        """
        Initialize a ROS-compatible UltraDES automaton node.
        """
        self.name = name
        self.role = role  # either "plant" or "supervisor"
        self.enabled_events_by_supervisors = {}

        if dfa_obj:
            self.automaton = dfa_obj
        elif automaton_path:
            self.automaton = load_from_json(automaton_path)
        else:
            raise ValueError("You must provide either 'automaton_path' or a DFA object.")

        self.transitions = transitions(self.automaton)
        self.current_state = initial_state(self.automaton)

        rospy.init_node(self.name, anonymous=False)

        self.pub_state = rospy.Publisher(f"/{self.name}/state", String, queue_size=10, latch=True)
        self.pub_events = rospy.Publisher(f"/{self.name}/possible_events", String, queue_size=10, latch=True)
        self.pub_marked = rospy.Publisher(f"/{self.name}/is_marked", String, queue_size=10, latch=True)

        if self.role == "supervisor":
            self.pub_enabled_events = rospy.Publisher(f"/{self.name}/enabled_events", String, queue_size=10, latch=True)

        rospy.sleep(1.0)

        self.sub_event = rospy.Subscriber("/event", String, self.event_callback)
        self.state_callbacks = self._create_state_callbacks()

        if self.role == "plant":
            self._discover_supervisors()

        self._enter_state(self.current_state)

    def _create_state_callbacks(self):
        """
        Create default state callbacks for each state.
        """
        callbacks = {}
        for state in states(self.automaton):
            state_name = str(state)

            def callback(s=state_name):
                rospy.loginfo(f"[{self.name}] Entered state '{s}' (default callback)")

            callbacks[state_name] = callback
        return callbacks

    def set_callback(self, state_name: str, callback_func):
        self.state_callbacks[state_name] = callback_func

    def _get_possible_events(self, state):
        """
        Return all events enabled from a given state.
        """
        return [str(event) for origin, event, dest in self.transitions if str(origin) == str(state)]

    def _get_filtered_events(self):
        """
        Return events allowed by all supervisors (intersection).
        """
        plant_events = set(self._get_possible_events(self.current_state))

        if self.role == "supervisor":
            rospy.loginfo(f"[{self.name}] Supervisor sees all events: {sorted(plant_events)}")
            return plant_events

        if not self.enabled_events_by_supervisors:
            return plant_events

        intersection = plant_events.copy()
        for events in self.enabled_events_by_supervisors.values():
            intersection &= set(events)

        return intersection

    def event_callback(self, msg):
        event = msg.data

        if event == "ping":
            rospy.loginfo(f"[{self.name}] Ping received — republishing state and events.")
            self._enter_state(self.current_state)
            return

        for origin, ev, dest in self.transitions:
            if str(origin) == str(self.current_state) and str(ev) == event:
                rospy.loginfo(f"[{self.name}] {origin} --({ev})--> {dest}")
                self.current_state = dest
                self._enter_state(self.current_state)
                return

        rospy.logwarn(f"[{self.name}] Invalid event '{event}' at state '{self.current_state}'")

    def _enter_state(self, state):
        rospy.loginfo(f"[{self.name}] Current state: {state}")
        self.pub_state.publish(str(state))

        events = self._get_filtered_events()
        rospy.loginfo(f"[{self.name}] Possible events: {events}")
        self.pub_events.publish(",".join(sorted(events)))

        if self.role == "supervisor":
            rospy.loginfo(f"[{self.name}] Enabled events: {sorted(events)}")
            self.pub_enabled_events.publish(",".join(sorted(events)))

        is_final = "True" if is_marked(state) else "False"
        rospy.loginfo(f"[{self.name}] Is marked state? {is_final}")
        self.pub_marked.publish(is_final)

        # Execute callback
        if str(state) in self.state_callbacks:
            self.state_callbacks[str(state)]()

    def _discover_supervisors(self):
        self.subscribed_supervisors = set()

        def check_new_supervisors(event):
            supervisors = rospy.get_param("/supervisors", [])
            new_supervisors = set(supervisors) - self.subscribed_supervisors

            for sup_name in new_supervisors:
                rospy.loginfo(f"[{self.name}] Subscribing to /{sup_name}/enabled_events")
                rospy.Subscriber(f"/{sup_name}/enabled_events", String, self._make_supervisor_callback(sup_name))
                self.subscribed_supervisors.add(sup_name)

        self.timer_supervisors = rospy.Timer(rospy.Duration(1.0), check_new_supervisors)

    def _make_supervisor_callback(self, sup_name):
        def callback(msg):
            events = msg.data.split(",") if msg.data else []
            rospy.loginfo(f"[{self.name}] Received events from {sup_name}: {events}")
            self.enabled_events_by_supervisors[sup_name] = events
            self._enter_state(self.current_state)
        return callback

    def run(self):
        rospy.spin()
