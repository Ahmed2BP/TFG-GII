#!/bin/bash

#envs=("Eplus-discrete-stochastic-hot-v1" "Eplus-discrete-stochastic-mixed-v1" "Eplus-discrete-stochastic-cool-v1" \
#"Eplus-continuous-stochastic-hot-v1" "Eplus-continuous-stochastic-mixed-v1" "Eplus-continuous-stochastic-cool-v1")

#envs=("Eplus-discrete-stochastic-hot-v1" "Eplus-discrete-stochastic-mixed-v1" "Eplus-discrete-stochastic-cool-v1")
# algorithms=("DDPG" "A2C" "DQN" "PPO" "RBC")
# envs=("Eplus-discrete-stochastic-hot-v1" "Eplus-discrete-stochastic-mixed-v1" "Eplus-discrete-stochastic-cool-v1")

# for alg in ${algorithms[*]}; do
#     for env in ${envs[*]}; do
#         docker exec -it a231f93867a5 /workspaces/sinergym/agents/algorithms/$alg.py -env $env -ep 2
#     done
# done

# docker exec -it a231f93867a5 /workspaces/sinergym/agents/algorithms/A2C.py -env Eplus-discrete-stochastic-mixed-v1 -ep 3

cd ../../
python /workspaces/sinergym/agents/algorithms/PPO.py -env Eplus-discrete-stochastic-mixed-v1 -ep 3