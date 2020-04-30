import tensorflow as tf
import numpy as np
import math


class Environment:

    def __init__(self):
        self.total_T = 1440  # 30s interval for 12h
        self.min_battery = 0.0
        self.max_battery = 200.0  # mAh
        self.event_prob = 0.04
        self.event_span = 10
        self.event_detect_reward = 1.0

        self.time = int(0)
        self.battery = self.max_battery    # np.random.uniform(self.min_battery, self.max_battery)
        self.event = False
        self.event_span_counter = None
        self.state = (self.battery, self.time, self.event)

    def step(self, action):
        state = self.state
        battery, time, event = state
        last_battery = battery

        # action_list = {'I', 'S', 'H'}
        if action == 0:
            battery += -10 * 30 / 3600
        elif action == 1:
            battery += -30 * 30 / 3600
        elif action == 2:
            battery += 75 * 30 / 3600

        battery = min(max(battery, self.min_battery), self.max_battery)
        time += 1

        if event == False and self.event_span_counter is None:
            ###### scenario A: event starts randomly with a given probability ######
            #event_judge = np.random.uniform(0, 1)
            #if event_judge < self.event_prob:
            #    self.event_span_counter = 0
            #    event = True
            ###### scenario B: event occurs regularly ######
            event_judge = bool((time - 1) % 58 == 0)   # event occurs once every 58 steps
            if event_judge == True:
                self.event_span_counter = 0
                event = True

        if event == True and self.event_span_counter <= self.event_span:
            self.event_span_counter += 1
            event = True
        if event == True and self.event_span_counter > self.event_span:
            self.event_span_counter = None
            event = False

        self.state = (battery, time, event)

        done = battery == self.min_battery or time == self.total_T
        done = bool(done)

        if not done:
            reward = 1.0    # each time the agent progresses one time step
            reward += 0.1 * (battery - last_battery)     # penalty for using battery, reward for harvesting energy
            if event == True and action == 1:    # if successfully detects event
                reward += self.event_detect_reward
        elif time < self.total_T - 1:
            reward = -10.0
        else:
            reward = 0.0

        return np.array(self.state), reward, event, done

    def reset(self):
        self.time = 0
        self.battery = self.max_battery   # np.random.uniform(self.min_battery, self.max_battery)
        self.event = False
        self.event_span_counter = None
        self.state = (self.battery, self.time, self.event)

        return np.array(self.state)