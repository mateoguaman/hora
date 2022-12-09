# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import ipdb
st = ipdb.set_trace
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_linear(nn.Module):
    def __init__(self, units, input_size):
        super(MLP_linear, self).__init__()
        layers = []
        for idx, output_size in enumerate(units):
            layers.append(nn.Linear(input_size, output_size))
            if idx < len(units) - 1:
                layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ProprioAdaptTConv(nn.Module):
    def __init__(self):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        self.horizon_length = kwargs.pop('horizon_length')
        self.train_mae = kwargs.pop('train_mae')
        mlp_input_shape = input_shape[0]
        # st()

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv()

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)


        if self.train_mae:
            self.timestep = torch.nn.Linear(1, out_size)
            self.preproc_x = MLP_linear(units=[out_size,out_size,out_size], input_size=out_size)
            self.postproc_x_action = MLP_linear(units=[out_size,out_size,actions_num], input_size=out_size)
            self.postproc_x_obs = MLP_linear(units=[out_size,out_size,actions_num*6], input_size=out_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        _, mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        # st()
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label
                extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return x, mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        # st()
        rst = self._actor_critic(input_dict)
        shared_feat, mu, logstd, value, extrin, extrin_gt = rst
        
        if self.train_mae:
            timestep_to_predict = np.expand_dims(input_dict['timestep_to_predict'],-1)
            # st()
            timestep_to_predict = torch.from_numpy(timestep_to_predict).float().to(input_dict['obs'].device)/self.horizon_length
            timestep_to_predict = self.timestep(timestep_to_predict)
            shared_feat_ = self.preproc_x(shared_feat)
            shared_feat_ = shared_feat_ + timestep_to_predict
            action_tgt_pred = self.postproc_x_action(shared_feat_)
            obs_tgt_pred = self.postproc_x_obs(shared_feat_)
            self.mse_loss = torch.nn.MSELoss()
            action_mse_loss = self.mse_loss(action_tgt_pred, input_dict['actions_target'])
            obs_mse_loss = self.mse_loss(obs_tgt_pred, input_dict['obs_target'])
            mae_loss = 0.5*action_mse_loss + obs_mse_loss
            # st()

        #     input_dict
        # st()
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        if self.train_mae:
            result['mae_loss'] = mae_loss
        return result
