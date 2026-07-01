# APRA-RL
A reinforcement learning (RL) simulation of agent bidding strategies in an ascending-price reimbursement auction.

OVERVIEW
This project studies a Deep Q Network Reinforcement Learning agent in the context of an ascending price reimbursement auction. The primary goal of this auction is to maximize auctioneer revenue and hype: a metric that studies the total bidding mass over time. The agent learns how to bid optimally against currently hardcoded opponents, leading to sometimes counterintuitive, but empirically optimal strategies.

TECHNOLOGIES
Current tech: Python, Pytorch (DQN), Gymnasium (Environment)
DQN chosen as the observation and action spaces are wide, and thus do not discretize too cleanly into a simple table. A neural network also better characterizes the "continous" nature of optimal bidding.

FILES
driver.py: parameter tuning, episodic and auction loops, data output
distinguished_agent.py: neural network brain for the distinguished agent, houses the learning and dual network logic
environment.py: auction environment instantiation and stepping, opponent bidding logic, reward function

KEY FINDINGS
* in progress; currently noting:
    - agent learned .2 round 1 overbid then abstain
    - facing a .2 overbid in round 1, completely abstain
        - forcing a larger bid in round 2 to win the round results in less reward

This project is currently a work in progress, with lots of testing and theory work being done

SETUP
* To be written here

ACKNOWLEDGEMENTS
This project is currently being contributed to by Jack Stanick and Miriam Nelson, with oversight from Francisco Marmolejo Cossio.