import json
import os
import sys
import ultrades
from ultrades.automata import dfa, state, event

# Optional: append UltraDES internal path if needed
sys.path.append(os.path.dirname(ultrades.__file__))

def load_from_json(path: str, name: str = "automaton"):
    """
    Load a DFA (Deterministic Finite Automaton) from a JSON file.

    The expected JSON file format:
    {
        "states": ["s0", "s1", ...],
        "events": ["e1", "e2", ...],
        "initial_state": "s0",
        "marked_states": ["s1"],
        "controllable_events": ["e1"],
        "transitions": {
            "s0": {"e1": "s1", "e2": "s0"},
            "s1": {"e1": "s0"}
        }
    }

    Args:
        path (str): Path to the JSON file.
        name (str): Optional name for the automaton.

    Returns:
        dfa: A DFA object compatible with UltraDES.
    """
    with open(path, "r") as file:
        data = json.load(file)

    state_map = {
        s: state(s, marked=(s in data.get("marked_states", [])))
        for s in data["states"]
    }

    event_map = {
        e: event(e, controllable=(e in data.get("controllable_events", [])))
        for e in data["events"]
    }

    transitions_list = []
    for origin, transitions_dict in data["transitions"].items():
        for ev, target in transitions_dict.items():
            transitions_list.append((state_map[origin], event_map[ev], state_map[target]))

    initial = state_map[data["initial_state"]]
    return dfa(transitions_list, initial, name)
