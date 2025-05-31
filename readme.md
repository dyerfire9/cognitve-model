# Clarion Cognitive Model Simulation

This is a cognitive simulation project that models skill acquisition in a typing task using the Clarion cognitive architecture. It compares two learning approaches: one with symbolic prior knowledge and one based on implicit trial-and-error learning.

## Features
- Simulates two learning systems: explicit (symbolic rules) vs. implicit (reinforcement-based)
- Tracks accuracy across 300 trials to measure learning performance
- Dynamically adds rules during simulation to reflect knowledge acquisition
- Visualizes learning curves for both systems using matplotlib
- Technologies: Python, pyClarion, matplotlib, pandas

## How to Run
Requirements
  - Python 3.8+
  - pip (Python package installer)

## Installation
In a terminal, navigate to the pyClarion folder then:

1. Clone the repository

git clone https://github.com/dyerfire9/cognitve-model.git
cd clarion-typing-simulation

2. Install as a regular library, run:
pip install .

Then run:
pip install matplotlib pandas

3. Run the simulation:
python model.py

4. Outputs:
- Output will be displayt in the terminal in an organized fashion
- A learning curve graph will be generated and displayed

## Note
- The simulation runs for 300 trials per condition by default
- Accuracy is measured incrementally to show learning progression
- You can modify the number of trials or learning threshold directly in `model.py`
- Rule definitions are either loaded from a `.ccml` file or generated dynamically


## Model Folder
The Model folder contains the coded model.
 
## Progress
- [x] Clarion agent with explicit knowledge
- [x] Implicit learning system - Implemented and tested logic of implicit learning system
- [ ] Accuracy tracking over trials
- [ ] Learning curve visualization & Graphs
- [ ] In Depth Analysis & Report


## Author information
Muhammad Abdul Mannan (corresponding author and repository maintainer) <br />
Student - University of Toronto <br />
LinkedIn: www.linkedin.com/in/abdulmannancomp <br />
Website:  https://dyerfire9.github.io/portfolio-site/ <br />
Email: abdulmannancomp@gmail.com <br />
