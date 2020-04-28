from environment import Environment
from agent import DQNAgent
import numpy as np
import io_plot as iop

EPISODES = 1000

if __name__ == "__main__":
    env = Environment()
    state_size = 2
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    output_episode = []

    # store for all episodes
    episode_total_detect_num = []
    episode_total_event_num = []
    episode_final_battery = []
    episode_total_reward = []
    episode_end_time = []

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        init_battery = state[0][0]
        event_num = 0
        detect_num = 0
        total_reward = 0

        step_battery = []
        step_detect_num = []
        step_event_num = []

        for time in range(env.total_T):
            action = agent.act(state)
            next_state, reward, event, done = env.step(action)
            if event == True:
                event_num += 1
                if action == 1:  # action_list = {'I', 'S', 'H'}
                    detect_num += 1
            total_reward += reward

            if e == 0 or e == EPISODES - 1:
                step_battery.append(state[0][0])
                step_detect_num.append(detect_num)
                step_event_num.append(event_num)

            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, event, action, reward, next_state, done)
            state = next_state
            if done:
                final_battery = state[0][0]
                print(
                    "episode: {}/{}, detect event: {}/{}, initial battery: {:.2f}, final battery: {:.2f}, end time: {}, total reward: {:.2f}"
                        .format(e, EPISODES, detect_num, event_num, init_battery, final_battery, time, total_reward))
                episode_total_detect_num.append(detect_num)
                episode_total_event_num.append(event_num)
                episode_final_battery.append(final_battery)
                episode_total_reward.append(total_reward)
                episode_end_time.append(time)
                if e == 0 or e == EPISODES - 1:
                    iop.save_to_file(step_battery, './data/step_battery.csv', 'a')
                    iop.save_to_file(step_detect_num, './data/step_detect_num.csv', 'a')
                    iop.save_to_file(step_event_num, './data/step_event_num.csv', 'a')
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    iop.save_to_file(episode_total_detect_num, './data/episode_total_detect_num.csv', 'w')
    iop.save_to_file(episode_total_event_num, './data/episode_total_event_num.csv', 'w')
    iop.save_to_file(episode_final_battery, './data/episode_final_battery.csv', 'w')
    iop.save_to_file(episode_total_reward, './data/episode_total_reward.csv', 'w')
    iop.save_to_file(episode_end_time, './data/episode_end_time.csv', 'w')

