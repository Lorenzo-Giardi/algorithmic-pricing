### Tabular Q-learning results

### RLlib Rollout script
1) Usage via terminal:
    ./rollout.py checkpoint-path --run algname --env envname --steps 1000 --out rollouts.pkl
    
2) Usage via Spyder
Add to command line options
checkpoint-path --run algname --env envname

--steps and --out are optional.

Example:
/home/lorenzo/ray_results/19_cont_DQN/DQN_MultiAgentFirmsPricingContinuous_1_2019-07-31_14-11-11sxig7ieu/checkpoint_3500/checkpoint-3500
 --run APEX --env firms_pricing_cont
