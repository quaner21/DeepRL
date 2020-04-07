import tensorflow as tf
import numpy as np
import math


class Environment:

    def __init__(self):
        self.total_T = 1440  # 30s interval for 12h
        self.min_battery = 0.0
        self.max_battery = 280.0  # mAh
        self.event_prob = 0.04
        self.event_span = 10
        self.event_detect_reward = 10.0

        self.action_space = ['I', 'S', 'H']

        self.time = int(0)
        self.battery = self.max_battery # np.random.uniform(self.min_battery, self.max_battery)
        self.event = False
        self.event_counter = None
        self.state = (self.battery, self.time)

    def step(self, action):
        state = self.state
        battery, time = state

        # action_list = {'I', 'S', 'H'}
        if action == 0:
            battery += -10 * 30 / 3600
        elif action == 1:
            battery += -30 * 30 / 3600
        elif action == 2:
            battery += 75 * 30 / 3600

        battery = min(max(battery, self.min_battery), self.max_battery)
        time += 1
        self.state = (battery, time)

        if self.event == False and self.event_counter is None:
            event_judge = np.random.uniform(0, 1)
            if event_judge < self.event_prob:
                self.event_counter = 0
                self.event = True
        if self.event == True and self.event_counter <= self.event_span:
            self.event_counter += 1
            self.event = True
        if self.event == True and self.event_counter > self.event_span:
            self.event_counter = None
            self.event = False

        done = battery == self.min_battery \
               or time == self.total_T
        done = bool(done)

        if not done:
            reward = 1.0
            if self.event == True and action == 1:
                reward += self.event_detect_reward
        else:
            reward = 0.0

        return np.array(self.state), reward, self.event, done

    def reset(self):
        self.time = 0
        self.battery = self.max_battery # np.random.uniform(self.min_battery, self.max_battery)
        self.event = False
        self.event_counter = None
        self.state = (self.battery, self.time)

        return np.array(self.state)