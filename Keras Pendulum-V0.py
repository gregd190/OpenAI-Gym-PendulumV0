# DNN Solution to OpenAI Gym's Pendulum-V0 environment
# Trains over 1000 episodes, plays 10 iterations and plots the reward rate vs number of iterations
# There is surely room for hyperparameter optimization, but it trains in minutes on a CPU and performs well.
# By default loads pre-trained weights, but code for training is commented out. 


import gym
import gym.spaces
import gym.wrappers
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque
from keras.layers import Flatten, Dense
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers

def create_action_bins(z):
	
	actionbins = np.linspace(-2.0, 2.0, z)
	
	return actionbins

def find_actionbin(action, actionbins):
    
    idx = (np.abs(actionbins - action)).argmin()
    
    return idx

def build_model(num_output_nodes):
	
	model = Sequential()
	
	model.add(Dense(128, input_shape = (3,), activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(num_output_nodes, activation = 'linear')) 
	
	adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999)
	
	model.compile(loss = 'mse', optimizer = adam)
	
	return model

def train_model(memory, gamma = 0.9):
	
	for state, actionbin, reward, state_new in memory:
		flat_state_new = np.reshape(state_new, [1,3])
		flat_state = np.reshape(state, [1,3])
		
		target = reward + gamma * np.amax(model.predict(flat_state_new))
			
		targetfull = model.predict(flat_state)
		
		targetfull[0][actionbin] = target
		
		model.fit(flat_state, targetfull, epochs = 1, verbose = 0) 
		
def run_episodes(eps = 0.999, r = False, iters = 100):
		
	eps_decay = 0.9999
	eps_min = 0.02
	
	for i in range(iters):
		
		state = env.reset()
	
		totalreward = 0
	
		memory = deque()
		
		cnt = 0	
		
		if eps>eps_min:
			eps = eps * eps_decay
		
		while cnt < 250:
			
			cnt += 1
			
			if r:
				env.render()
			if np.random.uniform() < eps:
				action = env.action_space.sample()
			else:
				flat_state = np.reshape(state, [1,3])
				action = np.amax(model.predict(flat_state))
				
			actionbin = find_actionbin(action, actionbinslist)
			
			action = actionbinslist[actionbin]
			action = np.array([action])
			
			observation, reward, done, _ = env.step(action)  
			
			totalreward += reward
			
			state_new = observation 
			
			memory.append((state, actionbin, reward, state_new))
			
			state = state_new
	
		train_model(memory, gamma = 0.9)
	
	return eps

#test without training
def play_game(eps = 0.0,  r = True):
	
	state = env.reset()
	totalreward = 0
	cnt = 0
	
	while cnt < 200:
		
		cnt += 1
		
		if r:
			env.render()
		if np.random.uniform() < eps:
			action = env.action_space.sample()
			actionbin = find_actionbin(action, actionbinslist)
		else:
			
			flat_state = np.reshape(state, [1,3])
			
			actionbin = np.argmax(model.predict(flat_state))
				
		action = actionbinslist[actionbin]
		action = np.array([action])
		
		observation, reward, done, _ = env.step(action)  
		
		totalreward += reward
		
		state_new = observation 
		
		state = state_new
		
	return totalreward

if __name__ == '__main__':

	env = gym.make('Pendulum-v0')
	
	eps = 1
	
	num_action_bins = 10
	
	actionbinslist = create_action_bins(num_action_bins)
	
	#Comment the following block if training from scratch
	
	print('loading model')
	model = load_model('pendulum-model.h5')
	print('model loaded')
	
	#Uncomment out the following section if training from scratch
	
	# ~ model = build_model(num_action_bins)
	
	# ~ totarray = []
	# ~ cntarray = []
	
	# ~ totaliters = 1000
	# ~ test_interval = 25	
	
	# ~ numeps = int(totaliters)
	
	# ~ print('numeps = ', numeps)
	
	# ~ cnt = 0
	
	# ~ while cnt < totaliters:
		
		# ~ eps = run_episodes(eps = eps, r = False, iters = test_interval)
	
		# ~ cnt += test_interval
		
		# ~ trarray = []
		# ~ for i in range(20):
			# ~ trarray.append(play_game(r = False))
		
		# ~ print(cnt, 'iterations. Average test reward = ', np.average(trarray))
		
		# ~ cntarray.append(cnt)
	
		# ~ totarray.append(np.average(trarray))
	
	
	# ~ #Plot average reward vs iterations
	# ~ plt.plot(cntarray, totarray)
	# ~ plt.xlabel('Iterations')
	# ~ plt.ylabel('Reward')
	# ~ plt.show()
	
	# ~ #Save model weights
	# ~ print('saving model')
	# ~ model.save('pendulum-model.h5')
	# ~ print('model saved')
	
	
	#Render 10 episodes to demonstrate performance		
	x = input('Hit enter to watch')
	for i in range(10):
		
		play_game(r = True)
		
	
	
	
	
	
	
		
	
		
