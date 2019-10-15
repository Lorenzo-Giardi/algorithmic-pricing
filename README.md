# Algorithmic Pricing
Environments and algorithms for studying the natural emergence of cooperation among multiple pricing algorithms driven by reinforcement learning and its implications for competition policy.

## Environments
Environments are contained in the folder "envs". Single agent environment are compatible with the OpenAi Gym interface, while multi-agent environments are compatible with that of Ray-RLlib. 
* Single agent prisoner dilemma (playing vs tit-for-tat opponent)
* Two agents prisoner dilemma
* N-agents firms pricing with discrete observation space
* N-agents firms pricing with continuous observation space

## Reinforcement learning algorithms
Algorithms and/or the scripts for their training are contained in the folder "train". 
* Tabular Q-learning
* Tabular Q-learning with Ray for parallel execution
* DQN with Ray-RLlib
The code for rolling out RLlib checkpoints is contained in the folder "rollout". \
The training and evaluation results are contained in the folder "train_results".


## Main requirements
* Python                        3.6
* tensorflow                    1.14.0 
* ray                           0.7.3
* gym                           0.14.0 
