import gym
import ray
import random

import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

env=MultiAgentPrisonerDilemma()

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
                        "agent_0": (None, env.observation_space, env.action_space, {}),
                        "agent_1": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn":
                    lambda agent_id:
                        random.choice(["agent_0", "agent_1"])
        },
})

    
for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
