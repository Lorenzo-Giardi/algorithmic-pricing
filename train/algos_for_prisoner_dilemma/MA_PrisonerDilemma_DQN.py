import gym
import ray
import random

import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

env=MultiAgentPrisonerDilemma()

def policy_mapping_fn(agent_id):
    if agent_id=="agent_0":
        return "policy_0"
    else:
        return "policy_1"

ray.shutdown()
ray.init()

"""
Trainers make algorithms accessible via Python API and command line, they manage
algorithm configuration, setup of the policy evaluators and optimizers,
and collection of training metrics. 

"""

trainer = dqn.DQNAgent(env=MultiAgentPrisonerDilemma, config={
        #"num_workers": 1,
        "num_envs_per_worker": 4,
        "num_cpus_per_worker": 4,
        "num_cpus_for_driver": 2,
        #"sample_batch_size": 200,
        #"train_batch_size": 200,
        "multiagent": {
                "policy_graphs": {
                        "policy_0": (None, env.observation_space, env.action_space, {}),
                        "policy_1": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn
        },
})

    
for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
