# DeepRL

Goal:
This project aims to optimize the energy use of an energy-harvesting IoT sensor, while maintaining its detection accuracy

Approach:
Deep Q-learning

Scenario:
State: {(battery, t)}
Action: {idle, sense, harvest}   
Time: with 30s interval, total time T=1440 (12h). One action per time step
Events: randomly created throughout T, duration of each event is a constant, total number of events is a constant
Battery: battery is fully charged at t=0 (max_battery)
Reward: 1 reward is given each time step. Extra rewards are given if events are successfully detected
