import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import math


"""

Variational inference for state space model
x_{t+1} ~ Dist(f(x_t,u_t))
y_{t}   ~ Dist(h(x_t))
y: observation
x: state
u: input

e.g. Normal distribution
x_{t+1}~ Normal(f_m(x_t,u_t),f_s(x_t,u_t))
y_{t}  ~ Normal(h(x_t),1)

"""
class VariationalStateSpaceModel(torch.nn.Module):
    def __init__(
        self,
        obs_dim,
        state_dim,
        input_dim,
        model_f,
        model_h,
        model_p0,
        model_q,
        model_q0,
        delta_t=0.1,
        alpha={},
        discrete_state=False,
        without_sampling=False,
        device=None,
    ):
        super(VariationalStateSpaceModel, self).__init__()
        self.delta_t = delta_t
        self.device=device

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.alpha=alpha
        self.model_h = model_h
        self.model_f = model_f
        self.model_p0 = model_p0
        self.model_q = model_q
        self.model_q0 = model_q0
        self.discrete_state=discrete_state
        self.without_sampling=without_sampling

    #posterior_estimation(obs, init_state, batch_size, step, input_)
    def posterior_estimation(self,obs, batch_size, step, input_=None, obs_mask=None):
        """
        Args:
            obs: batch_size x #step x obs_dim
            input_: batch_size x #step x input_dim
        Returns:
            state: batch_size x (0 - step-1) x state_dim
            state_dist_q: batch_size x (0 - step-1) x state_dim
            state_dist_p: batch_size x (1 - step-1) x state_dim
        """
        state = []
        state_dist_q = []
        ###
        ### Estimating initial state
        ###
        dist_q0=self.model_q0((obs, input_, obs_mask))
        if self.without_sampling:
            current_state = dist_q0.mean
        else:
            current_state = dist_q0.rsample()
        state.append(current_state)
        state_dist_q.append(dist_q0)
        ###
        ### Estimating posterior states
        ###
        state_dist_p = []
        for t in range(step-1):
            ###
            ### computing q(next state | current state ...)
            ###
            next_dist_q=self.model_q((obs,current_state, input_, obs_mask, t+1))
            ### qs ~ q(...)
            if self.without_sampling:
                next_state_qs=next_dist_q.mean
            else:
                next_state_qs=next_dist_q.rsample()
            ###
            ### computing p(next state | current state, input)
            ###
            if input_ is None:
                next_dist_p=self.model_f((current_state, None))
            else:
                next_dist_p=self.model_f((current_state, input_[:, t, :]))
            ###
            state.append(next_state_qs)
            state_dist_q.append(next_dist_q)
            state_dist_p.append(next_dist_p)
            current_state = next_state_qs
        ### sampled state
        state = torch.stack(state, dim=1)
        return state, state_dist_q, state_dist_p
    def filtering(self, obs, input_):
        p0=self.model_p0()
        current_particle=p0.sample(100)
        for t in range(1,step):
            ###
            ### computing p(next state | current state, input)
            ###
            if input_ is None:
                next_dist_p=self.model_f((current_particle, None))
            else:
                next_dist_p=self.model_f((current_particle, input_[:, t, :]))
            #
            ps=next_dist_p.mean
            obs_generated=self.model_h(ps)
            w = obs_generated.log_prob(obs)
            reample_dist=torch.nn.distributions.Categorical(w)
            idx=reample_dist.saple(100)
            new_particle=ps[idx]
            current_particle=new_particle

    def simulate_one_step(self, state_t):
        next_dist_p=self.model_f((state_t, None))
        return next_dist_p.mean

    def forward(self, obs, input_, obs_mask=None):
        """
        Args:
            obs: batch_size x #step x obs_dim
            input_: batch_size x #step x input_dim
        Returns:
            loss: dictionary
            state: batch_size x (0 - step-1) x state_dim
            obs_generated: batch_size x #step x obs_dim
        """
        step = obs.shape[1]
        batch_size = obs.shape[0]
        ###
        ### computing probabilities of states 
        ### q(z_{t+1}| z_{t}, x_{1:T}, u_{1:T})
        ### p(z_{t+1}| z_{t}, u_{t})
        ###
        state,state_dist_q,state_dist_p = self.posterior_estimation(obs, batch_size, step, input_, obs_mask)
        obs_generated=self.model_h(state)
        ###
        ### observation loss: negative log likelihood
        ###
        nll=-obs_generated.log_prob(obs)
        if obs_mask is not None:
            nll=nll*obs_mask
        loss_recons = torch.sum(nll,dim=(1,2)) #torch.sum((obs - obs_generated) ** 2,dim=-1)
        loss_sum_recons=loss_recons.mean(dim=0)
        
        ### verbose length
        #print("q",len(state_dist_q)) => step
        #print("p",len(state_dist_p)) => step-1
        #print("sample:",state.shape)

        ###
        ### kl loss: t=0
        ###
        p0=self.model_p0()
        q0=state_dist_q[0]
        if self.discrete_state:
            kl0_loss=torch.distributions.kl.kl_divergence(torch.distributions.OneHotCategorical(q0.logits),torch.distributions.OneHotCategorical(p0.logits))
        else:
            kl0_loss=torch.distributions.kl.kl_divergence(q0,p0)
            kl0_loss=torch.sum(kl0_loss,dim=1)
        kl0_loss=torch.mean(kl0_loss,dim=0)
        ###
        ### kl loss: t=1:T
        ###
        kl_t=[]
        for t in range(1,step):
            q=state_dist_q[t]
            p=state_dist_p[t-1]
            if self.discrete_state:
                kl=torch.distributions.kl.kl_divergence(torch.distributions.OneHotCategorical(q.logits),torch.distributions.OneHotCategorical(p.logits))
            else:
                kl=torch.distributions.kl.kl_divergence(q,p)
                kl=torch.sum(kl,dim=1)
            kl_t.append(kl)
        kl_loss=torch.sum(torch.stack(kl_t,dim=1),dim=1)
        kl_loss=torch.mean(kl_loss,dim=0)
        ###
        ### summation loss
        ###
        loss = {
            "recons": self.alpha["recons"]*loss_sum_recons,
            "*recons": loss_sum_recons,
            "kl": self.alpha["temporal"]*kl_loss,
            "*kl": kl_loss,
            "kl0": self.alpha["temporal"]*kl0_loss,
            "*kl0": kl0_loss,
        }
        return loss, state, obs_generated.mean


