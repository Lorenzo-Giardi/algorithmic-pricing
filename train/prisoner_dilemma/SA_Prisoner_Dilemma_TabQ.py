import gym
import random
import matplotlib.pyplot as plt

"""
Based on:
https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
https://github.com/adventuresinML/adventures-in-ml-code/blob/master/r_learning_python.py
"""

def s_id(arg):
    if arg.all() == np.array([0, 0]).all():
        return 0
    elif arg.all() == np.array([0, 1]).all():
        return 1
    elif arg.all() == np.array([1, 0]).all():
        return 2
    elif arg.all() == np.array([1, 1]).all():
        return 3
    else:
        raise ValueError('Problem in the state to id loop')

def eps_greedy_tabular_q_learning(env,num_episodes=500):
    q_table = np.zeros((4,2))
    gamma = 0.95
    eps = 0.9
    lr = 0.05
    decay_factor = 0.99
    progress_rew = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        state = env.reset()
        eps *= decay_factor
        done = False
        cumul_rew = 0
        
        while not done:
            if np.random.random() < eps or np.sum(q_table[s_id(state),:])==0:
                action = np.random.randint(0,2)
            else:
                action = np.argmax(q_table[s_id(state),:])
            new_state, reward, done, _ = env.step(action)
            q_table[s_id(state),action] += reward + lr * (
                    gamma * np.max(q_table[s_id(new_state),:]) - q_table[s_id(state),action])
            state = new_state
            cumul_rew += reward
            
        progress_rew[i] = cumul_rew 
    
        
    return q_table, progress_rew

q_table, progress_rew = eps_greedy_tabular_q_learning(RepeatedPrisonerDilemma(),num_episodes=500)
plt.plot(progress_rew)
