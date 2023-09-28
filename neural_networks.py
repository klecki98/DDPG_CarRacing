import os
import numpy as np
import torch as T
import  torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, hidden_dims, n_actions,
                 name, checkpoint_dir='tmp'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, name)

        neuron_list = [input_dims]+hidden_dims

        self.layers_list = []
        self.batch_norms_list = []

        for i in range(len(neuron_list)-1):
            self.layers_list.append(nn.Linear(neuron_list[i], neuron_list[i+1]))
            self.batch_norms_list.append(nn.LayerNorm(neuron_list[i+1]))

        self.layers_list = nn.ModuleList(self.layers_list)
        self.batch_norms_list = nn.ModuleList(self.batch_norms_list)

        for i in range(len(self.layers_list)):
            amp = 1./np.sqrt(self.layers_list[i].weight.data.size()[0])
            self.layers_list[i].weight.data.uniform_(-amp,amp)
            self.layers_list[i].bias.data.uniform_(-amp,amp)

        self.action_value = nn.Linear(self.n_actions, neuron_list[-1])
        amp_action = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-amp_action,amp_action)
        self.action_value.bias.data.uniform_(-amp_action, amp_action)

        self.q = nn.Linear(neuron_list[-1], 1)
        amp_q = 0.003
        self.q.weight.data.uniform_(-amp_q,amp_q)
        self.q.bias.data.uniform_(-amp_q, amp_q)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation, action):
        temp = observation
        for i in range(len(self.layers_list)):
            temp = self.layers_list[i](temp)
            temp = self.batch_norms_list[i](temp)
            if i < (len(self.layers_list)-1):
                temp = F.relu(temp)
        action_temp = self.action_value(action)
        temp = F.relu(T.add(temp, action_temp))
        temp = self.q(temp)

        return temp

    def save(self):
        print("saving_critic")
        T.save(self.state_dict(), self.checkpoint)

    def load(self):
        print("loading_critic")
        self.load_state_dict(T.load(self.checkpoint))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, hidden_dims, n_actions,
                 name, checkpoint_dir='tmp'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, name)

        neuron_list = [input_dims]+hidden_dims

        self.layers_list = []
        self.batch_norms_list = []

        for i in range(len(neuron_list)-1):
            self.layers_list.append(nn.Linear(neuron_list[i], neuron_list[i+1]))
            self.batch_norms_list.append(nn.LayerNorm(neuron_list[i+1]))

        self.layers_list = nn.ModuleList(self.layers_list)
        self.batch_norms_list = nn.ModuleList(self.batch_norms_list)

        for i in range(len(self.layers_list)):
            amp = 1./np.sqrt(self.layers_list[i].weight.data.size()[0])
            self.layers_list[i].weight.data.uniform_(-amp,amp)
            self.layers_list[i].bias.data.uniform_(-amp,amp)

        self.mu = nn.Linear(neuron_list[-1], self.n_actions)
        amp_mu = 0.003
        self.mu.weight.data.uniform_(-amp_mu,amp_mu)
        self.mu.bias.data.uniform_(-amp_mu, amp_mu)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        temp = observation
        for i in range(len(self.layers_list)):
            temp = self.layers_list[i](temp)
            temp = self.batch_norms_list[i](temp)
            temp = F.relu(temp)
        temp = T.tanh(self.mu(temp))
        return temp

    def save(self):
        print("saving_actor")
        T.save(self.state_dict(), self.checkpoint)

    def load(self):
        print("loading_actor")
        self.load_state_dict(T.load(self.checkpoint))

