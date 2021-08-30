import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import math

class ModelH(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[8,8], activation=F.leaky_relu):
        super(ModelH, self).__init__()
        linears=[]
        prev_d=state_dim
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)

        self.out_mu = self.get_layer_out(prev_d,obs_dim)
        self.activation = activation

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        return l
    def get_layer_out(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.xavier_normal_(l.weight)
        return l



    def forward(self, x):
        res_x=x
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.activation(x)
        m=self.out_mu(x)
        return torch.distributions.Normal(m,1.0)


class ModelF(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[3], activation=F.leaky_relu):
        super(ModelF, self).__init__()
        self.input_dim = input_dim
        linears=[]
        prev_d=state_dim
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)

        self.out_mu = self.get_layer(prev_d, state_dim)
        self.out_sigma = self.get_layer(prev_d, state_dim)
        self.activation = activation

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        return l

    def forward(self, x):
        current_state, input_=x
        x=current_state
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.activation(x)
        if False:
            m=self.out_mu(x)
            s=torch.clip(torch.nn.functional.softplus(self.out_sigma(x)),1.0e-10,1.0)
            return torch.distributions.Normal(m,s)
        else:
            m=self.out_mu(x)
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)


class PositionalEncoding0(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, d_model, max_len)
        pd = torch.t(position * div_term)
        pe[0, 0::2, :] = torch.sin(pd)
        pe[0, 1::2, :] = torch.cos(pd)
        self.register_buffer('pe', pe)
        self.max_len=max_len

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim, seq_len]
        """
        s=self.max_len//2-t
        #x = x + self.pe[:,:,s:s+x.size(2)]
        x = x * self.pe[:,:,s:s+x.size(2)]
        #return self.dropout(x)

        #y = torch.tile(self.pe[:,:,s:s+x.size(2)],(x.size(0),1,1))
        #x = torch.cat([x,y],dim=1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        #(1, 1, max_len)
        #pe = torch.exp(-torch.abs(torch.arange(0, max_len, 1)-max_len/2))
        #pe=pe.unsqueeze(dim=0).unsqueeze(dim=0)
        pe = torch.zeros(1, 1, max_len)
        pe[:,:,max_len//2]=1
        self.register_buffer('pe', pe)
        self.max_len=max_len

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim, seq_len]
        """
        s=self.max_len//2-t
        x = x*self.pe[:,:,s:s+x.size(2)]
        #y = torch.tile(self.pe[:,:,s:s+x.size(2)],(x.size(0),1,1))
        #x = torch.cat([x,y],dim=1)
        return x

class PositionalEncoding2(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        #(1, 1, max_len)
        pe = torch.exp(-torch.abs(torch.arange(0, max_len, 1)-max_len/2))
        pe=pe.unsqueeze(dim=0).unsqueeze(dim=0)
        self.w = torch.nn.Parameter(pe)
        #nn.init.kaiming_uniform_(self.w)
        #self.pe = torch.nn.Parameter(torch.ones(1,1,max_len))
        self.max_len=max_len

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim, seq_len]
        """
        s=self.max_len//2-t
        pe=torch.nn.functional.softmax(self.w,dim=-1)
        #pe=self.w
        x = x*pe[:,:,s:s+x.size(2)]
        #y = torch.tile(self.pe[:,:,s:s+x.size(2)],(x.size(0),1,1))
        #x = torch.cat([x,y],dim=1)
        return x



class ModelQ(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[3], activation=F.leaky_relu):
        super(ModelQ, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(obs_dim+input_dim, 8, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(8, 8, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(8, 8, kernel_size=3, padding='same')

        self.rnn = nn.LSTM(obs_dim+input_dim, 8, num_layers=1 , batch_first=True)
        
        linears=[]
        prev_d=state_dim+8
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)

        self.out_mu = self.get_layer(prev_d, state_dim)
        self.out_sigma = self.get_layer(prev_d, state_dim)
        self.activation = activation
        self.pos_encoder = PositionalEncoding(d_model=8)
        self.padding=torch.nn.ConstantPad1d(3, 0)


    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        return l

    def forward(self, x):
        obs,current_state, input_, t=x
        if self.input_dim ==0:
            x1=obs
        else:
            x1=torch.cat([obs,input_],dim=1)
        x2=current_state
        ###

        
        ###
        ### RNN type
        ###
        xt=x1[:,:t+1,:]
        o,_ = self.rnn(xt)
        x1=o[:,t,:]
        

        """
        ###
        ### CNN type 1
        ###
        x1=x1.permute(0,2,1)
        x1=self.padding(x1)
        pad=3
        x1=x1[:,:,t-3+pad:t+3+pad]
        x1 = self.activation(self.conv1(x1))
        x1 = self.activation(self.conv2(x1))
        x1 = torch.sum(x1,dim=2)
        """

        """
        ###
        ### CNN type 2
        ###
        x1=x1.permute(0,2,1)
        x1 = F.activation(self.conv1(x1))
        x1 = F.activation(self.conv2(x1))
        x1 = F.activation(self.conv3(x1))
        x1 = self.pos_encoder(self.conv4(x1),t)
        x1 = torch.sum(x1,dim=2)
        #x1=x1[:,:,t]
        """
        ###
        x=torch.cat([x1,x2],dim=1)
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.activation(x)
        if False:
            m=self.out_mu(x)
            s=torch.clip(torch.nn.functional.softplus(self.out_sigma(x)),1.0e-10,1.0)
            return torch.distributions.Normal(loc=m,scale=s)
        else:
            m=self.out_mu(x)
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)

class ModelP0(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim):
        super(ModelP0, self).__init__()
        self.mu    = torch.nn.Parameter(torch.zeros(state_dim))
        self.sigma = torch.nn.Parameter(torch.ones(state_dim))

    def forward(self):
        if False:
            s=torch.clip(self.sigma,1.0e-10,1.0)
            return torch.distributions.Normal(self.mu,s)
        else:
            m=self.mu
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)



class ModelQ0(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[3], activation=F.leaky_relu):
        super(ModelQ0, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(obs_dim+input_dim, 8, 2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 8, 2)
        
        linears=[]
        prev_d=8
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)

        self.out_mu = self.get_layer(prev_d, state_dim)
        self.out_sigma = self.get_layer(prev_d, state_dim)
        self.activation = activation

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        return l

    def forward(self, x):
        obs, input_=x
        if self.input_dim ==0:
            x1=obs
        else:
            x1=torch.cat([obs,input_],dim=1)
        x1=x1.permute(0,2,1)
        ###
        x1 = self.pool(self.activation(self.conv1(x1)))
        x1 = self.pool(self.activation(self.conv2(x1)))
        x1 = torch.sum(x1,dim=2)
        ###
        x=x1
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.activation(x)
        if False:
            m=self.out_mu(x)
            s=torch.clip(torch.nn.functional.softplus(self.out_sigma(x)),1.0e-10,1.0)
            return torch.distributions.Normal(m,s)
        else:
            m=self.out_mu(x)
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)



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
        model_v,
        model_v0,
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
        self.model_v = model_v
        self.model_v0 = model_v0
        self.discrete_state=discrete_state
        self.without_sampling=without_sampling

    #posterior_estimation(obs, init_state, batch_size, step, input_)
    def posterior_estimation(self,obs, batch_size, step, input_=None):
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
        if input_ is None:
            dist_q0=self.model_v0((obs, None))
        else:
            dist_q0=self.model_v0((obs, input_))
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
            if input_ is None:
                next_dist_q=self.model_v((obs,current_state, None, t+1))
            else:
                next_dist_q=self.model_v((obs,current_state, input_, t+1))
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
        
    def forward(self, obs, input_):
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
        state,state_dist_q,state_dist_p = self.posterior_estimation(obs, batch_size, step, input_)
        obs_generated=self.model_h(state)
        ###
        ### observation loss: negative log likelihood
        ###
        nll=-obs_generated.log_prob(obs)
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


