import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random



# REINFROCE Network
class REINFORCE(nn.Module):
    def __init__(self, state_space=None,
                       action_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(REINFORCE, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"
        

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(state_space, hidden_dim))

        # Add hidden layers
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, action_space))

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = F.softmax(self.layers[-1](x), dim=0)

        return out

def train(model, roll_out, optimizer, gamma, device):
    G = 0

    optimizer.zero_grad()

    for r, prob in roll_out[::-1]:
        G = r + gamma * G
        loss = -torch.log(prob) * G
        loss.to(device).backward()
    optimizer.step()
    
def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # Determine seeds
    model_name = "REINFORCE"
    env_name = "CartPole-v1"
    seed = 1
    exp_num = 'SEED_'+str(seed)

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default 'log_dir' is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+"_"+model_name+"_"+exp_num)

    # set parameters
    learning_rate = 0.0001
    episodes = 5000
    print_per_iter = 100
    max_step = 20000
    discount_rate = 0.99

    Policy = REINFORCE(state_space=env.observation_space.shape[0],
                       action_space=env.action_space.n,
                       num_hidden_layer=2,
                       hidden_dim=64).to(device)
    
    

    # Set Optimizer
    optimizer = optim.SGD(Policy.parameters(), lr=learning_rate)

    for epi in range(episodes):
        s = env.reset()
        done = False

        roll_out = []
        step = 0

        # Set score
        score = 0

        while (not done) and (step < max_step) :
            # if epi%print_per_iter == 0:
            #     env.render()

            # Get action
            a_prob = Policy(torch.from_numpy(s).float().to(device))
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            roll_out.append((r, a_prob[a]))

            s = s_prime
            score += r
            step += 1
        
        
        train(Policy, roll_out, optimizer, discount_rate, device)

        # Logging
        print("epsiode :{}, score :{}".format(epi, score))
        writer.add_scalar('Rewards per epi', score, epi)
        save_model(Policy, model_name+"_"+".pth")

    writer.close()
    env.close()
