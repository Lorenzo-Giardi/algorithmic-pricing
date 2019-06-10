import gym      
import ray
import random

### USING TRAINER

import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

trainer = dqn.DQNAgent(env='custom_envs:repeatedprisoner-v0', config={
        #"num_workers": 1,
        "num_envs_per_worker": 4,
        "num_cpus_per_worker": 4,
        "num_cpus_for_driver": 2,
        #"sample_batch_size": 200,
        #"train_batch_size": 200,
        })

    
for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        
### USING TUNE  

from ray import tune

ray.shutdown()
ray.init()

tune.run(
        "DQN",
        stop={"training_iteration": 400},
        config={"env": 'custom_envs:repeatedprisoner-v0',
                "num_gpus": 0,
                "num_workers": 4,
                "num_envs_per_worker": 8,
                },
)
