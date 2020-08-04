"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append("")

from backend.torch.distributions import Normal
from backend.policies.base import Policy
from backend.torch import pytorch_util as ptu
from backend.torch.core import PyTorchModule
from backend.torch.data_management.normalizer import TorchFixedNormalizer
from backend.torch.modules import LayerNorm

LOG_SIG_MAX = 5
LOG_SIG_MIN = -20


def identity(x):
    return x

'''
@File    :   latentgnn_v1.py
@Time    :   2019/05/27 13:39:43
@Author  :   Songyang Zhang 
@Version :   1.0
@Contact :   sy.zhangbuaa@hotmail.com
@License :   (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University
@Desc    :   None
'''

class LatentGNNV1(PyTorchModule):
    """
    Latent Graph Neural Network for Non-local Relations Learning
    Args:
        latent_dims (list): List of latent dimensions  
        channel_stride (int): Channel reduction factor. Default: 4
        num_kernels (int): Number of latent kernels used. Default: 1
        without_residual (bool): Flag of use residual connetion. Default: False
        norm_layer (nn.Module): Module used for batch normalization. Default: nn.BatchNorm2d.
        norm_func (function): Function used for normalization. Default: F.normalize
    """
    def __init__(self, 
                input_dim, 
                latent_dims, 
                output_size,
                num_kernels=1,
                hidden_dict = {'in':[64, 64],
                               'out':[32, 32]},
                norm_layer=nn.BatchNorm2d, 
                norm_func=F.normalize,
                ):

        self.save_init_params(locals())
        super(LatentGNNV1, self).__init__()

        self.num_kernels = num_kernels
        self.norm_func = norm_func
        self.input_size = input_dim
        self.output_size = output_size

        # Define the latentgnn kernel
        assert len(latent_dims) == num_kernels, 'Latent dimensions mismatch with number of kernels'

        for i in range(num_kernels):
            self.add_module('LatentGNN_Kernel_{}'.format(i), 
                                LatentGNN_Kernel(input_dim=input_dim, 
                                                num_kernels=num_kernels,
                                                hidden_layers = hidden_dict,
                                                latent_dim=latent_dims[i],
                                                norm_layer=norm_layer,
                                                norm_func=norm_func))

        # Residual Connection
        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.kernel_channel = nn.Sequential(
                                    nn.Conv1d(in_channels=num_kernels, 
                                            out_channels=1,
                                            kernel_size=1, padding=0, bias=False)
        )
        
        self.latent_mapping = FlattenMlp(input_size=input_dim,
                                        hidden_sizes = [],
                                        output_size=output_size,
                                        bias = False)

    def forward(self, feature):
        # Generate visible space feature 
        out_features = []
        for i in range(self.num_kernels):
            out_features.append(eval('self.LatentGNN_Kernel_{}'.format(i))(feature))
        
        out_features = torch.cat(out_features, dim=2) if self.num_kernels > 1 else out_features[0]
        out_features = self.kernel_channel(out_features.permute(0,2,1))

        out_features = self.latent_mapping(out_features.squeeze(0))

        return out_features    


class LatentGNN_Kernel(PyTorchModule):
    """
    A LatentGNN Kernel Implementation
    Args:
    """
    def __init__(self, input_dim, latent_dim, hidden_layers, num_kernels,
                        norm_layer,
                        norm_func):
        self.save_init_params(locals())
        super(LatentGNN_Kernel, self).__init__()
#         super().__init__()
        self.norm_func = norm_func
        #----------------------------------------------
        # Step 1 & 3: Visible-to-Latent & Latent-to-Visible
        #----------------------------------------------
        self.psi_in = FlattenMlp(
            hidden_sizes = hidden_layers['in_'],
            input_size=input_dim,
            output_size=latent_dim,
            bias = False
        )
        
        self.psi_out = FlattenMlp(
            hidden_sizes = hidden_layers['out_'],
            input_size=input_dim,
            output_size=1,
            bias = False
        )

    def forward(self, feature):

        #----------------------------------------------
        # Step1 : Contexts-to-Latent 
        #----------------------------------------------
#         print(feature.shape)
        phi = self.psi_in(feature)
#         print(phi.shape)
        graph_adj_in = F.softmax(phi, dim=1)
#         print(graph_adj_in.shape)
        latent_node_feature = torch.bmm(graph_adj_in.permute(0,2,1), feature)
#         print(latent_node_feature.shape)

        #----------------------------------------------
        # Step2 : Latent-to-Latent 
        #----------------------------------------------
        # Generate Dense-connected Graph Adjacency Matrix
        latent_node_feature_n = self.norm_func(latent_node_feature, dim=-1)
#         print(latent_node_feature_n.shape)
        affinity_matrix = torch.bmm(latent_node_feature_n, latent_node_feature_n.permute(0,2,1))
#         print(affinity_matrix.shape)
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)
#         print(affinity_matrix.shape)
        latent_node_feature = torch.bmm(affinity_matrix, latent_node_feature)
#         print(affinity_matrix.shape)
        
        #----------------------------------------------
        # Step3: Latent-to-Output
        #----------------------------------------------
        graph_adj_out = F.softmax(self.psi_out(latent_node_feature), dim = 1)
#         print(graph_adj_out.shape)
        output = torch.bmm(latent_node_feature.permute(0,2,1), graph_adj_out)
        
#         print(output.shape)
        
        return output


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            bias = True
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.bias = bias
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size
        self.use_GNN = False

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size, bias = bias)
            in_size = next_size
            hidden_init(fc.weight)
            if self.bias is not False:
                fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)
       
class NormalAux(Mlp):
    def __init__(
                self,
                hidden_sizes,
                input_size,
                output_size,
                std=None,
                init_w=5e-3,
                **kwargs
        ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=input_size,
            output_size=output_size,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = input_size
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, output_size)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    @torch.no_grad()
    def get_beliefs(self, deterministic=False):
        # outputs = self.forward(obs, deterministic=deterministic)[0]
        raise NotImplementedError

    def forward(self, 
                z,
                task_descriptor = None,
                reparameterize=False,
                deterministic=False,
                return_log_prob=False,
                aux_no_grad = False,
                ):
        if aux_no_grad:
            h = z.detach()
        else:
            h = z
        for _, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        log_prob = None
        expected_log_prob = None

        if deterministic:
            belief = mean
        else:
            normal = Normal(mean, std)
            if return_log_prob:
                if reparameterize:
                    belief = normal.rsample()
                else:
                    belief = normal.sample()
                log_prob = normal.log_prob(task_descriptor)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    belief = normal.rsample()
                else:
                    belief = normal.sample()

        return (
            belief, mean, log_std, log_prob, expected_log_prob, std
        )

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''
    def reset(self, num_tasks=1):
        pass


class LatentGNNEncoder(LatentGNNV1):
    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.use_GNN = False
        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)




