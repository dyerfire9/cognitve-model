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

def build_agent(stimuli_features, action_config, rules_path=None, use_rules=True):
    """
    Builds a Clarion agent with optional explicit rules.
    """
    with cl.Structure("agent") as agent:
        cl.Module("input", cl.Receptors(stimuli_features))
        cl.Module("params", cl.Repeat(), ["params"])
        cl.Module("null", cl.Repeat(), ["null"])
        with cl.Structure("acs"):
            cl.Module("bi", cl.CAM(), ["../input"])
            cl.Module("bu", cl.BottomUp(), ["fr_store#0", "fr_store#1", "fr_store#2", "bi"])
            cl.Module("fr", cl.ActionRules(), ["../params", "fr_store#3", "fr_store#4", "bu"])
            cl.Module("td", cl.TopDown(), ["fr_store#0", "fr_store#1", "fr#0"])
            cl.Module("bo", cl.CAM(), ["td"])
            cl.Module("act", cl.ActionSampler(), ["../params", "bo"], ["../act#cmds"])
            cl.Module("fr_store", cl.Store(), ["../params", "../null", "../null", "../null"])
        cl.Module("act", cl.Actions(action_config), ["acs/act#0"])

    # Parameters for rule selection and action choice
    agent["params"].output = cl.NumDict({
        cl.feature("acs/fr#temp"): 0.01,
        cl.feature("acs/act#temp"): 0.01
    })

    # Leave rule loading for later (will add this next)
    return agent

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