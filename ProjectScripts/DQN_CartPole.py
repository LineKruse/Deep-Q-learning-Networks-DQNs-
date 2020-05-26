#  A pole is attached by an un-actuated joint to a cart,
#  which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.
#  The pendulum starts upright, and the goal is to prevent it from falling over.
#  A reward of +1 is provided for every timestep that the pole remains upright.
#  The episode ends when the pole is more than 15 degrees from vertical,
#  or the cart moves more than 2.4 units from the center.


import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import os
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, Conv2D,MaxPooling2D
from numpy import array
from keras.models import Model

env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.lr = 0.001
        self.memory = deque(maxlen=10000)
        self.model = self.build_model()

    X = (env.observation_space.shape[0],64)
    X = np.expand_dims(X, axis=2)

    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []
    episodes = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        episode = e
        state = env.reset()
        state = np.reshape(state, (1, 4))
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 4))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        episodes.append(episode)
    
    #Save csv file with loss per episode
    os.chdir('/Users/lineelgaard/Documents/Data Science/ExamProject - DQN Transfer Learning')
    data = np.column_stack((loss, episodes))
    savetxt('lossCartPole_8_7.csv', data, delimiter=',')

    # serialize model to JSON
    model = agent.model
    model_json = model.to_json()
    os.chdir('/Users/lineelgaard/Documents/Data Science/ExamProject - DQN Transfer Learning')
    with open("modelCartPole.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelCartPole.h5")
    print("Saved model to disk")
    
    return loss
    

def random_policy(episode, step):
    loss = []
    episodes = []
    for i_episode in range(episode):
        env.reset()
        episode = i_episode
        score = 0
        print(i_episode)
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
            #print("Starting next episode")
        loss.append(score)
        episodes.append(episode)

    os.chdir('/Users/lineelgaard/Documents/Data Science/ExamProject - DQN Transfer Learning')
    data = np.column_stack((loss, episodes))
    savetxt('lossCartPole_7_1.csv', data, delimiter=',')


if __name__ == '__main__':

    episode = 200
    loss = train_dqn(episode) #run DQN model 
    #loss = random_policy(episode, 1000) #run random policy
    plt.plot([i+1 for i in range(0, episode, 2)], loss[::2])
    plt.show()
