from environment import Environment
from agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

EPISODES = 1000

if __name__ == "__main__":
    env = Environment()
    state_size = 2
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        init_battery = state[0]
        event_num = 0
        success_detect = 0
        for time in range(1440):
            action = agent.act(state)
            next_state, reward, event, done = env.step(action)
            if event == True:
                event_num += 1
                if action == 1:  # action_list = {'I', 'S', 'H'}
                    success_detect += 1
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, event, action, reward, next_state, done)
            state = next_state
            if done:
                final_battery = state[0]
                print("episode: {}/{}, detect event: {}/{}, initial battery: {:.2}, final battery: {:.2}, end time: {}, e: {:.2}"
                      .format(e, EPISODES, success_detect, event_num, init_battery[0], final_battery[0], time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
