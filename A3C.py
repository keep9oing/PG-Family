import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp


import numpy as np
import random
import time

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


def train(global_Actor, global_Critic, device, rank):

    env = gym.make(env_name)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.n

    np.random.seed(seed+rank)
    random.seed(seed+rank)
    seed_torch(seed+rank)
    env.seed(seed+rank)

    local_Actor = Actor(state_space=env_state_space,
                  action_space=env_action_space,
                  num_hidden_layer=2,
                  hidden_dim=64).to(device)
    local_Critic = Critic(state_space=env_state_space,
                    num_hidden_layer=2,
                    hidden_dim=64).to(device)

    local_Actor.load_state_dict(global_Actor.state_dict())
    local_Critic.load_state_dict(global_Critic.state_dict())

    batch = []

    # Set Optimizer
    actor_optimizer = optim.Adam(global_Actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(global_Critic.parameters(), lr=critic_lr)

    for epi in range(episodes):
        s = env.reset()

        done = False
        score = 0
        
        step = 0
        while (not done) and (step < max_step):

            # Get action
            a_prob = local_Actor(torch.from_numpy(s).float().to(device))
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            done_mask = 0 if done is True  else 1

            batch.append([s,r/100,s_prime,a_prob[a],done_mask])
            
            if len(batch) >= batch_size:
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

                    v_s = local_Critic(s_buf)
                    v_prime = local_Critic(s_prime_buf)

                    Q = r_buf+discount_rate*v_prime.detach()*done_buf # value target
                    A =  Q - v_s                              # Advantage
                    
                    # Update Critic
                    critic_optimizer.zero_grad()
                    critic_loss = F.mse_loss(v_s, Q.detach())
                    critic_loss.backward()
                    for global_param, local_param in zip(global_Critic.parameters(), local_Critic.parameters()):
                        global_param._grad = local_param.grad
                    critic_optimizer.step()

                    # Update Actor
                    actor_optimizer.zero_grad()
                    actor_loss = 0
                    for idx, prob in enumerate(prob_buf):
                        actor_loss += -A[idx].detach() * torch.log(prob)
                    actor_loss /= len(prob_buf) 
                    actor_loss.backward()

                    for global_param, local_param in zip(global_Actor.parameters(), local_Actor.parameters()):
                        global_param._grad = local_param.grad
                    actor_optimizer.step()

                    local_Actor.load_state_dict(global_Actor.state_dict())
                    local_Critic.load_state_dict(global_Critic.state_dict())

                    batch = []

            s = s_prime
            score += r
            step += 1
        
    env.close()
    print("Process {} Finished.".format(rank))



def test(global_Actor, device, rank):
    
    env = gym.make(env_name)

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    for epi in range(episodes):
        s = env.reset()

        done = False
        score = 0
        step = 0 
        
        while (not done) and (step < max_step):
            # Get action
            a_prob = global_Actor(torch.from_numpy(s).float().to(device))
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            s = s_prime
            score += r
            step += 1

        print("EPISODES:{}, SCORE:{}".format(epi, score))

    env.close()
        


def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


# Global variables
model_name = "Actor-Critic"
env_name = "CartPole-v1"
seed = 1
exp_num = 'SEED_'+str(seed)

# Global parameters
actor_lr = 1e-4
critic_lr = 1e-3
episodes = 5000
print_per_iter = 100
max_step = 20000
discount_rate = 0.99
batch_size = 5

if __name__ == "__main__":
    # Set gym environment
    env = gym.make(env_name)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.n

    if torch.cuda.is_available():
        device = torch.device("cuda")

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    global_Actor = Actor(state_space=env_state_space,
                  action_space=env_action_space,
                  num_hidden_layer=2,
                  hidden_dim=64).to(device)
    global_Critic = Critic(state_space=env_state_space,
                    num_hidden_layer=2,
                    hidden_dim=64).to(device)
    
    env.close()

    global_Actor.share_memory()
    global_Critic.share_memory()

    processes = []
    process_num = 6

    mp.set_start_method('spawn') # Must be spawn
    print("MP start method:",mp.get_start_method())

    for rank in range(process_num):
        if rank == 0:
            p = mp.Process(target=test, args=(global_Actor, device, rank, ))
        else:
            p = mp.Process(target=train, args=(global_Actor, global_Critic, device, rank, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()