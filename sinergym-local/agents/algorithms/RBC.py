#!/usr/bin/python3

import gym
import argparse
import numpy as np
import mlflow

import sys
sys.path.append("../../")

from sinergym.utils.controllers import RBC5Zone
from sinergym.utils.wrappers import LoggerWrapper


import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--environment', '-env', type=str, default=None)
parser.add_argument('--episodes', '-ep', type=int, default=1)
args = parser.parse_args()

environment = args.environment
n_episodes = args.episodes
env = gym.make(environment)
env = LoggerWrapper(env)

name = 'RBC-' + environment + '-' + str(n_episodes) + '-episodes'

with mlflow.start_run(run_name=name):

    # create rule-based agent
    agent = RBC5Zone(env)

    setpoints = []   # Array to store temperatures

    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
        while not done:
            a = agent.act(obs)
            setpoints.append(a)     # Store setpoint in array
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:
                current_month = info['month']
                print(info['month'], sum(rewards))
        print('Episode ', i, 'Mean reward: ', np.mean(rewards), 'Cumulative reward: ', sum(rewards))
    env.close()

    setpoints = np.array(setpoints)

    # Plotting historic of setpoints
    fig = plt.figure()
    plt.title("Evolución de setpoints")
    plt.xlabel("Timesteps")
    plt.ylabel("Temperatura setpoints")

    plt.plot(setpoints[:,0], label="Setpoints para calefacción")
    plt.plot(setpoints[:,1], "r-", label="Setpoints para refrigeración")
    plt.legend()
    plt.show()

    unique, counts = np.unique(setpoints, return_counts=True, axis=0)
    print("Setpoints\t Veces utilizados")
    for i in range(len(unique)):
        print(str(unique[i]) + "\t " + str(counts[i]))


    mlflow.log_metric('mean_reward', np.mean(rewards))
    mlflow.log_metric('cumulative_reward', sum(rewards))

    mlflow.end_run()