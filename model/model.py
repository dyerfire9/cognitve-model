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
import pandas

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
def run_typing_task(agent, stimuli_list, trials=300, learn=False, error_threshold=5):
    """
    Run a typing task simulation with the given agent.
    :param agent: the Clarion agent (with or without prior rules).
    :param stimuli_list: list of stimulus identifiers (e.g., ["letter-A", "letter-B", "letter-C"]).
    :param trials: number of trials to simulate.
    :param learn: if True, enable trial-and-error learning (for implicit condition).
    :param error_threshold: number of errors on a letter before adding an explicit rule (implicit learning).
    :return: (accuracy_record, correct_count) over the trials.
    """
    input_module = agent["input"]       # reference to input module
    action_module = agent["act"]       # reference to action (output) module
    correct_count = 0
    accuracy_record = []
    # Dictionary to count errors per stimulus (used only if learn=True)
    error_count = {s: 0 for s in ["A", "B", "C"]}
    
    for i in range(trials):
        # 1. Present a random letter stimulus
        stimulus_feature = random.choice(stimuli_list)     # e.g., "letter-A"
        input_module.process.stimulate([stimulus_feature]) # activate the input for that letter

        # 2. Advance the agent one decision step
        agent.step()

        # 3. Observe the agent's chosen action (chunk with highest activation)
        if action_module.output:
            chosen_action = max(action_module.output, key=action_module.output.get)
        else:
            chosen_action = cl.chunk("no_action")  # Placeholder chunk
        # Determine the correct expected action chunk for this stimulus
        letter = stimulus_feature.split("-")[1]            # get "A", "B", or "C"
        correct_action_chunk = cl.feature("act#type", f"press_{letter.lower()}") # e.g., cl.chunk('press_a')
        # Check if the agent's response was correct
        is_correct = (chosen_action == correct_action_chunk)
        correct_count += int(is_correct)
        
        # 4. Provide feedback and (if implicit learning) update knowledge
        if learn:
            if not is_correct:
                # Increment error count for this letter stimulus
                error_count[letter] += 1
                # If errors reach threshold for this letter, add an explicit rule for it
                if error_count[letter] == error_threshold:
                    # Construct a CCML rule to map this letter to the correct action
                    rule_ccml = f"""store acs/fr_store:
    ruleset learned:
        rule:
            conc:
                act#cmd-type press_{letter.lower()}
            cond:
                input#letter-{letter}
"""
                    cl.load(io.StringIO(rule_ccml), agent)
                    # After this, the agent has effectively learned the rule for this letter
                    # (future responses for this letter will likely be correct)
        
        # Record running accuracy up to this trial
        accuracy = correct_count / (i + 1)
        accuracy_record.append(accuracy)
        # Optionally print progress every 50 trials or at the end
        if (i + 1) % 50 == 0 or i == trials - 1:
            print(f"Trial {i+1:3d}: Stimulus = {stimulus_feature}, Chosen Action = {chosen_action}, ",
                  f"Correct = {is_correct}, Accuracy = {accuracy:.2%}")
    return accuracy_record, correct_count

# Main Execution for Both Conditions
def main():
    # Define the task stimuli and actions
    letters = ["A", "B", "C"]
    stimuli_features = [f"letter-{s}" for s in letters]         # e.g., ["letter-A", "letter-B", ...]
    action_config = {"type": [f"press_{s.lower()}" for s in letters]}  # {"type": ["press_a","press_b","press_c"]}
    total_trials = 300
    
    # Condition 1: With prior explicit knowledge (rules given)
    print("\n===== CONDITION 1: EXPLICIT SYSTEM (With Prior Knowledge) =====")
    agent_explicit = build_agent(stimuli_features, action_config, rules_path="typing_rules.ccml", use_rules=True)
    acc_with, corr_with = run_typing_task(agent_explicit, stimuli_features, trials=total_trials, learn=False)
    
    # Condition 2: Without prior knowledge (implicit learning only)
    print("\n===== CONDITION 2: IMPLICIT SYSTEM (No Prior Knowledge) =====")
    agent_implicit = build_agent(stimuli_features, action_config, use_rules=False)
    acc_without, corr_without = run_typing_task(agent_implicit, stimuli_features, trials=total_trials, learn=True, error_threshold=5)
    
    # Calculate performance metrics
    # Trials to reach 90% accuracy (if never reached, use total_trials)
    trials_to_90_with = next((i+1 for i, acc in enumerate(acc_with) if acc >= 0.90), total_trials)
    trials_to_90_without = next((i+1 for i, acc in enumerate(acc_without) if acc >= 0.90), total_trials)
    # Total errors made (out of total_trials)
    errors_with = total_trials - corr_with
    errors_without = total_trials - corr_without
    print(f"\nResults with prior knowledge: reached 90% accuracy by trial {trials_to_90_with}, total errors = {errors_with}.")
    print(f"Results without prior knowledge: reached 90% accuracy by trial {trials_to_90_without}, total errors = {errors_without}.")

    # Plot learning curves for accuracy over time
    plt.figure(figsize=(6,4))
    plt.plot(acc_with, label="With Prior Knowledge", color="green")
    plt.plot(acc_without, label="Without Prior Knowledge", color="red")
    plt.title("Typing Task Accuracy Over 300 Trials")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()