# Transfer learning using deep Q-learning networks (DQNs). 
DQNs combine Q-learning algorithms with neural network architecture to approximate optimal policy on reinforcement-learning (RL) problems. 
The code here was used to test whether a DQN trained on the Cartpole game would learn faster on a new game environment, the Lunar Lander game, compared to an algorithm with no prior training, i.e., whether DQNs facilitates transfer learning on RL problems. 

Transfer learning is tested under different conditions
- varying the amount of network layers with pre-trained weights 
- varying the position of transfer layers in network architecture 
- varying amounts of hidden layers in DQN 

Best transfer learning performance was obtained with a D3-DQN applying pre-trained weights in teh second hidden layer and reinitializing wegihts in the thrid layer. 
