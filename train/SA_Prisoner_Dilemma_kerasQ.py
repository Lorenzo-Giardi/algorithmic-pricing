import gym
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import matplotlib.pyplot as plt

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
 

def keras_q_learning(env,num_episodes=500):
    
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1,4)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
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
        if i%20 ==0:
            print('Episode {} of {}'.format(i+1, num_episodes))
        
        while not done:
            if np.random.random() < eps:
                action = np.random.randint(0,2)
            else:
                action = np.argmax(model.predict(np.identity(4)[s_id(state):s_id(state)+1]))
            
            new_state, reward, done, _ = env.step(action)
            
            target = reward + gamma * np.max(model.predict(np.identity(4)[s_id(new_state):s_id(new_state)+1]))
            target_vec = model.predict(np.identity(4)[s_id(state):s_id(state)+1])[0] 
            target_vec[action] = target
            model.fit(np.identity(4)[s_id(state):s_id(state)+1], target_vec.reshape(-1,2), epochs=1, verbose=0)
            
            state = new_state
            cumul_rew += reward
            
        progress_rew[i] = cumul_rew 
    
    progress_ep = np.array(range(num_episodes))
        
    return progress_rew, progress_ep

progress_rew, progress_ep = keras_q_learning(RepeatedPrisonerDilemma(),num_episodes=500)

plt.plot(progress_ep, progress_rew)
