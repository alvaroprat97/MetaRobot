import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain = 1)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, torch.nn.Tanh):
        torch.nn.init.xavier_uniform_(m_weight, gain = 1.4)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(self.obs_dim + self.action_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)

        # self.apply(weights_init)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = F.layer_norm(F.relu(self.linear1(state_action)), [self.hidden_size])
        x = F.layer_norm(F.relu(self.linear2(x)), [self.hidden_size])
        x = F.relu(self.linear3(x))

        return x


class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_size, max_action = None):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_action = torch.FloatTensor(max_action)

        self.linear1 = nn.Linear(self.obs_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.action_dim)

        # self.apply(weights_init)

    def forward(self, state):

        x = F.layer_norm(F.relu(self.linear1(state)), [self.hidden_size])
        x = F.layer_norm(F.relu(self.linear2(x)), [self.hidden_size])
        x = torch.tanh(self.linear3(x))
        # print(x.shape)
        # print(self.max_action.shape)
        shape = x.shape
        x = x*self.max_action
        # assert shape == x.shape
        # print(x.shape)
        return x
