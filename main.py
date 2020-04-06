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
        for time in range(1440):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
