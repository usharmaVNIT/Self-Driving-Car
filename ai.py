#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:04 2020

@author: Official
"""


import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable



class Network(nn.Module):
# self is used to refer to the object that is being created
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # If you want 1 Hidden layer you need 2 connections. Now these connections can be full 
        # or partial now first layer consists of input and hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        #second connection is b/w hidden Layer (30) to output layer
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        #hidden neurons activated from rectifier function
        # Now by applying the function it will give back values
        x = F.relu(self.fc1(state))
        #Now feed these values in the 2nd connection
        q_values = self.fc2(x)
        return q_values
    
    
    
# Now We will implement Experience Learning by making batches of experiences
#   Now suppose our capacity is 100 then an experience will be learnet from 
#   100 transitions
        
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        # create a memory for capacity==>int events
        self.memory = []
    
    def push(self, event):
        # this event is a tupleof 4 elements laststate St,  new state St+1, last action At, last reward rt
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self, batch_size):
        # if list = ((1,2,3),(4,5,6)) zip(*list) = ((1,4),(2,5),(3,6))
        samples = zip(*random.sample(self.memory, batch_size))
        # therefore if memory is ((s1,s2,a1,r1),(s2,s3,a2,r2)) ==> ((s1,s2),(s2,s3),(a1,a2),(r1,r2))
        # concatinate wrt 1st dimension cat(x,0). 0 is first dimension
        return map(lambda x: Variable(torch.cat(x,0)), samples)
    

# Implementing Deep Q Learning
        # here in this class we will use the network and experience replay therefore we need all the 
    #   arguments in init for network ans replatmemory
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        # create a reward window that will store mean of rewards
        self.reward_window = []
        # Now To create a neural Network
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        # we will now create an optimizer and Adam is good enough
        # Now Adam will require parameters like learning,etc and all parameters of our model.
        # Note that learning rate should not be high as we want our 
        # A.I to properly learn
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # Now we will keep track of last state , reward , action
        # our state consist of 5 or input_size values that is 3 sensors ,orentation , -orentation
        # but for pytorch needs more than a vector a torch tensor and it also needs to have a fake dimension called batch
        # as the neural network will take a batch of vectors that is batch dimension
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    
    def select_action(self, state):
        probabilities = F.softmax(self.model(Variable(state, volatile = True))*10)  # T=10
        # Note that state here is a pytorch tensor coming from last_state which is a tensor
        # therefore model will accept it but we dont need the gradient therefore we use the Variable
        # method and make it volatile to remove the gradient. You can also do without it try
        # Now for a softmax function we need a temprature coff and in this case it is 10
        # if softmax((1,2,3)) = (0.04,0.11,0.85) then softmax((1,2,3)*3) = (0,0.02,0.98)
        # Now we have to choose a random action grom a sample having these probabilities
        # and we do it by random 
        action = probabilities.multinomial(1)
        #prob.multi returns the pytorch variable with a fake dimension therefore we need to change it.
        return action.data[0,0]
    # Note if you want to deactivate the ai put T =0
    
    def learn(self , batch_state, batch_next_state, batch_reward, batch_action):
        # Note these arguments are markovs decision process framework (st,s(t+1),rt,a)
        # they are all aligned w.r.t time thanks to the concatination we made in sample w.r.t first dimension
        # output = self.model(batch_state)
        # we now get the output of the model but there another technical trick, here we will get all the possible actions but
        # we are interested in only the action that were decided by the network to play at each time and to do this
        # we use a predefined function gather and we only want 1 action
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Note this will output q value
        # here in this unsqueez we used 1 instead of 0 as 0 is the fake dimension of states and 1 is the fake dimension of 
        # action. Now by this we will have our outputs in a batch but we want them in a simple tensor or vector
        # the batch is needed by the neural networks but we dont need them as we have our outputs
        # and in the balance equation we dont need them in a batch so we remove(kill) the batch(fake dimension)
        # therrfore we squeeze it w.r.t action(1);
        next_output = self.model(batch_next_state).detach().max(1)[0]
        # Now in computing the target we use the eqn rt+ (gamma)*max Q(a,st+1))
        # Now i detach every output and take the max w.r.t the action performed, i.e. 1 ,
        # according to the state, i.e. 0,.
        target = self.gamma*next_output + batch_reward;
        # Now we compute the temporal difference loss
        # Now ew use a predefined function hoober loss
        # it requres two input that is our prediction and the target
        td_loss = F.smooth_l1_loss(outputs, target)
        # Now we will use back propogation and use stocastic gradient
        # We will use the optimizer we defined above
        # we reinitialize the optimizer at each iteration as to give correct result and it is done by
        # zero_grad
        self.optimizer.zero_grad()
        td_loss.backward()
        # The purpose of this retain variable is to free some memory 
        # as we will go through this many times and will improve traiing performance
        # Now we will update the weights
        self.optimizer.step()
        
    # It will return the action on reaching the next state
    def update(self, reward, signal):
        new_state = torch.Tensor(signal).float().unsqueeze(0)
        # You need to wrap it in 1 list or add another dimension 1*size to represent it as a state 
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.LongTensor([self.last_reward])))
        # Note all of them need to ba a pytorch tensor and we have converted do so
        action = self.select_action(new_state)
        # Now after performing an action it needs to learn from it
        # Now ew need to learn from past 100 expriences
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            # Now to call learn from the object we do self
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(self.last_reward)
        # Now we ensure that reward window is of constant size
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action
    
    # Now we will implement the function that will calculate the mean of reward
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1) # to avoid 0 in denominator
    
    # Now we will save the brain and therefore we will save the weights and the the optimizer
    def save(self):
        torch.save({'state_dict' : self.model.state_dict(),'optimizer' : self.optimizer.state_dict},'last_braintst.pth')
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint ......")
            checkpoint = torch.load('last_braintst.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print(" !! No Checkpoint found !!")
            