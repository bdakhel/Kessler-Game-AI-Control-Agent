# Kessler-Game-Controller
A Kessler Game Controller made using the concepts of fuzzy systems and optimized by genetic algorithms

A fuzzy control agent to play the Kessler implementation of the Asteroids arcade game, version 1.3.6, written by Thales North America as a part of the Explainable Fuzzy Challenge (XFC) competition at the NAFIPS annual conference. The agent must at a minimum determine the thrust, turn_rate, and fire control outputs using a fuzzy controller; a genetic algorithm optimizes the fuzzy system.

To install the required libraries:

pip install kesslergame==1.3.6
pip install scikit-fuzzy
pip install EasyGA

To simulate controllers:
`python3 scenario_test.py`

# See the controller in action!
Our optimized controller (green) outperforms a controller that simply shoots targets
https://github.com/bdakhel/Kessler-Game-Controller/assets/90705800/03af748b-b516-433d-80b2-0dead40c8095

