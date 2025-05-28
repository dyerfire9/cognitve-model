# Author: Muhammad Abdul Mannan
# This model simulates a simple typing skill acquisition task (pressing A, B, C)
# under two conditions: 
# 1) With prior explicit knowledge (pre-defined letter-to-key rules).
# 2) Without prior knowledge (implicit trial-and-error learning).
# The agent is based on the CLARION architecture, using pyClarion for implementation.
# We track the agent's accuracy over 300 trials and measure learning speed and errors.
import pyClarion as cl
import random
import io
import matplotlib.pyplot as plt

# Agent Construction Function
def build_agent(stimuli_features, action_config, rules_path=None, use_rules=True):
    """
    Construct a Clarion agent for the typing task.
    :param stimuli_features: list of stimulus feature identifiers (e.g., ["letter-A", ...]).
    :param action_config: dictionary defining action chunks (e.g., {"type": ["press_a", ...]}).
    :param rules_path: path to a CCML file containing explicit production rules (if any).
    :param use_rules: whether to load prior explicit rules into the agent.
    :return: a pyClarion agent structure.
    """
    with cl.Structure("agent") as agent:
        # Input module (Receptors) for letter stimuli
        cl.Module("input", cl.Receptors(stimuli_features))
        # Parameter module (Repeat) to hold global parameters (like softmax temperature)
        params = cl.Module("params", cl.Repeat(), ["params"])
        # Null module (Repeat) as a dummy input (used to pad inputs where needed)
        cl.Module("null", cl.Repeat(), ["null"])
        # Action-centered subsystem (acs) structure
        with cl.Structure("acs"):
            # Bottom-level input to implicit action network
            cl.Module("bi", cl.CAM(), ["../input"])  # CAM: Chunk Assemblage Module
            # Bottom-up integration of implicit and any learned explicit info
            cl.Module("bu", cl.BottomUp(), ["fr_store#0", "fr_store#1", "fr_store#2", "bi"])
            # Action rule module (explicit rules influence) combining params, rule store, and bottom-up
            cl.Module("fr", cl.ActionRules(), ["../params", "fr_store#3", "fr_store#4", "bu"])
            # Top-down module to propagate chosen rule effects back (for completeness)
            cl.Module("td", cl.TopDown(), ["fr_store#0", "fr_store#1", "fr#0"])
            # Second-stage chunk network combining top-down effects
            cl.Module("bo", cl.CAM(), ["td"])
            # Action selection (chooses an action command based on combined inputs)
            cl.Module("act", cl.ActionSampler(), ["../params", "bo"], ["../act#cmds"])
            # Rule store module to hold fixed/learned rules (with four input slots as required)
            cl.Module("fr_store", cl.Store(), ["../params", "../null", "../null", "../null"])
        # Actions module defining actual output actions (chunks) for pressing keys
        cl.Module("act", cl.Actions(action_config), ["acs/act#0"])
    
    # Set softmax temperature parameters for rule selection and action selection 
    # (low temperature => makes the choice nearly deterministic)
    params.output = cl.NumDict({
        cl.feature("acs/fr#temp"): 0.01,   # temperature for rule selection (explicit)
        cl.feature("acs/act#temp"): 0.01   # temperature for action selection (implicit/overall)
    })
    
    # Load symbolic rules for letter-to-key mapping if prior knowledge is used
    if use_rules:
        # Try loading from a CCML file if provided; otherwise use an inline definition
        if rules_path is not None:
            try:
                with open(rules_path) as f:
                    cl.load(f, agent)  # load explicit production rules into fr_store
            except FileNotFoundError:
                # Define inline CCML rules if file not found
                rules_ccml = """store acs/fr_store:
    ruleset typing:
        rule:
            conc:
                act#cmd-type press_a
            cond:
                input#letter-A
        rule:
            conc:
                act#cmd-type press_b
            cond:
                input#letter-B
        rule:
            conc:
                act#cmd-type press_c
            cond:
                input#letter-C
"""
                cl.load(io.StringIO(rules_ccml), agent)
        else:
            # If no path given but use_rules is True, load from a default rule set string
            rules_ccml = """store acs/fr_store:
    ruleset typing:
        rule:
            conc:
                act#cmd-type press_a
            cond:
                input#letter-A
        rule:
            conc:
                act#cmd-type press_b
            cond:
                input#letter-B
        rule:
            conc:
                act#cmd-type press_c
            cond:
                input#letter-C
"""
            cl.load(io.StringIO(rules_ccml), agent)
    # If use_rules is False, we do not load any explicit knowledge (start from scratch)
    return agent

# Simulation Function
def run_typing_task():
    print('test')

def main():
    letters = ["A", "B", "C"]
    stimuli_features = [f"letter-{s}" for s in letters]
    action_config = {"type": [f"press_{s.lower()}" for s in letters]}
    
    print("Running typing task with prior knowledge (explicit rules)...")
    agent = build_agent(stimuli_features, action_config, use_rules=True)
    
    acc_record, correct_total = run_typing_task(agent, stimuli_features, trials=150)
    
    print(f"Final Accuracy: {correct_total / 150:.2%}")

if __name__ == "__main__":
    main()