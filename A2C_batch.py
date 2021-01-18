import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(Critic, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(state_space, hidden_dim))

        # Add hidden layers
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = self.layers[-1](x)

        return out

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_space=None,
                       action_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(Actor, self).__init__()

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

def train(actor, critic, 
          critic_optimizer, actor_optimizer,
          gamma,
          batch,
          device):

    s_buf = []
    s_prime_buf = []
    r_buf = []
    prob_buf = []
    done_buf = []

    for item in batch:
        s_buf.append(item[0])
        r_buf.append(item[1])
        s_prime_buf.append(item[2])
        prob_buf.append(item[3])
        done_buf.append(item[4])

    s_buf = torch.FloatTensor(s_buf).to(device)
    r_buf = torch.FloatTensor(r_buf).unsqueeze(1).to(device)
    s_prime_buf = torch.FloatTensor(s_prime_buf).to(device)
    done_buf = torch.FloatTensor(done_buf).unsqueeze(1).to(device)

    v_s = critic(s_buf)
    v_prime = critic(s_prime_buf)

    Q = r_buf+gamma*v_prime.detach()*done_buf # value target
    A =  Q - v_s                      # Advantage

    # Update Critic
    critic_optimizer.zero_grad()
    critic_loss = F.mse_loss(v_s, Q.detach())
    critic_loss.backward()
    critic_optimizer.step()

    # Update Actor
    actor_optimizer.zero_grad()
    actor_loss = 0
    for idx, prob in enumerate(prob_buf):
        actor_loss += -A[idx].detach() * torch.log(prob)
    actor_loss /= len(prob_buf) 
    actor_loss.backward()
    actor_optimizer.step()

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # Determine seeds
    model_name = "Actor-Critic"
    env_name = "LunarLander-v2"
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
    actor_lr = 1e-4
    critic_lr = 1e-3
    episodes = 5000
    print_per_iter = 100
    max_step = 20000
    discount_rate = 0.99

    batch = []
    batch_size = 5 # 5 is best until now

    critic = Critic(state_space=env.observation_space.shape[0],
                    num_hidden_layer=2,
                    hidden_dim=64).to(device)
    
    actor = Actor(state_space=env.observation_space.shape[0],
                  action_space=env.action_space.n,
                  num_hidden_layer=2,
                  hidden_dim=64).to(device)
    

    # Set Optimizer
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)

    for epi in range(episodes):
        s = env.reset()

        done = False
        score = 0
        

        step = 0
        while (not done) and (step < max_step):
            # if epi%print_per_iter == 0:
            #     env.render()

            # Get action
            a_prob = actor(torch.from_numpy(s).float().to(device))
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            done_mask = 0 if done is True  else 1

            batch.append([s,r/100,s_prime,a_prob[a],done_mask])
            
            if len(batch) >= batch_size:
                train(actor, critic, 
                    critic_optimizer, actor_optimizer, 
                    discount_rate,
                    batch,
                    device)
                batch = []

            s = s_prime
            score += r
            step += 1


        # Logging
        print("epsiode :{}, score :{}".format(epi, score))
        writer.add_scalar('Rewards per epi', score, epi)
        save_model(actor, model_name+"_"+".pth")

    writer.close()
    env.close()