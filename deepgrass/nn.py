import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import math

class ModelH(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[16,16], activation=F.leaky_relu):
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
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[16], activation=F.leaky_relu,discrete_state=False):
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
        self.discrete_state=discrete_state

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
        if not self.discrete_state:
            m=current_state+self.out_mu(x)*0.01
            s=torch.clip(torch.nn.functional.softplus(self.out_sigma(x)),1.0e-10,1.0)
            m=torch.clip(m,-2.0,2.0)
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
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[16,16], activation=F.leaky_relu,discrete_state=False, obs_mask_enabled=False):
        super(ModelQ, self).__init__()
        self.input_dim = input_dim
        if obs_mask_enabled:
            in_dim = 2*obs_dim+input_dim
        else:
            in_dim = obs_dim+input_dim

        self.conv1 = nn.Conv1d(in_dim, 8, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(8, 8, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(8, 8, kernel_size=3, padding='same')

        self.rnn = nn.LSTM(in_dim, 8, num_layers=1 , batch_first=True)
        
        linears=[]
        prev_d=state_dim+8
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)

        self.out_mu = self.get_layer(prev_d, state_dim,True)
        self.out_sigma = self.get_layer(prev_d, state_dim,True)
        self.activation = activation
        self.pos_encoder = PositionalEncoding(d_model=8)
        self.padding=torch.nn.ConstantPad1d(3, 0)
        self.discrete_state=discrete_state
        self.obs_mask_enabled=obs_mask_enabled


    def get_layer(self,in_d,out_d,linear_flag=False):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        if not linear_flag:
            nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        else:
            nn.init.normal_(l.weight,0.0,0.01)
        return l

    def forward(self, x):
        obs, current_state, input_, obs_mask, t=x
        if self.input_dim ==0:
            x1=obs
        else:
            x1=torch.cat([obs,input_],dim=-1)
        if self.obs_mask_enabled:
            x1=torch.cat([x1,obs_mask],dim=-1)
        x2=current_state
        ###

        """
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
        ###
        ### CNN type 2
        ###
        x1=x1.permute(0,2,1)
        x1 = F.activation(self.conv1(x1))
        x1 = F.activation(self.conv2(x1))
        x1 = F.activation(self.conv3(x1))
        x1 = self.pos_encoder(self.conv4(x1),t)
        x1 = torch.sum(x1,dim=2)
        x1=x1[:,:,t]
        """
        ###
        x=torch.cat([x1,x2],dim=1)
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.activation(x)
        if not self.discrete_state:
            m=self.out_mu(x)
            s=torch.clip(torch.nn.functional.softplus(self.out_sigma(x)),1.0e-10,1.0)
            m=torch.clip(m,-2.0,2.0)
            return torch.distributions.Normal(loc=m,scale=s)
        else:
            m=self.out_mu(x)
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)

class ModelP0(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim,discrete_state=False):
        super(ModelP0, self).__init__()
        self.mu    = torch.nn.Parameter(torch.zeros(state_dim)+torch.normal(0,0.01,size=(state_dim,)))
        self.sigma = torch.nn.Parameter(torch.ones(state_dim)*0.5+torch.normal(0,0.01,size=(state_dim,)))
        self.discrete_state=discrete_state
    def forward(self):
        if not self.discrete_state:
            s=torch.clip(self.sigma,1.0e-10,1.0)
            return torch.distributions.Normal(self.mu,s)
        else:
            m=self.mu
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)



class ModelQ0(torch.nn.Module):
    def __init__(self, obs_dim, state_dim, input_dim, h_dim=[16,16], activation=F.leaky_relu,discrete_state=False, obs_mask_enabled=False):
        super(ModelQ0, self).__init__()
        self.input_dim = input_dim
        if obs_mask_enabled:
            in_dim = 2*obs_dim+input_dim
        else:
            in_dim = obs_dim+input_dim

        self.conv1 = nn.Conv1d(in_dim, 8, 2)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 8, 2)
        
        linears=[]
        prev_d=8
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)

        self.out_mu = self.get_layer(prev_d, state_dim,True)
        self.out_sigma = self.get_layer(prev_d, state_dim,True)
        self.activation = activation
        self.discrete_state=discrete_state
        self.obs_mask_enabled=obs_mask_enabled

    def get_layer(self,in_d,out_d,linear_flag=False):
        l=nn.Linear(in_d, out_d)
        #nn.init.kaiming_uniform_(l.weight)
        if not linear_flag:
            nn.init.kaiming_normal_(l.weight,nonlinearity="leaky_relu")
        else:
            nn.init.normal_(l.weight,0.0,0.01)

        return l

    def forward(self, x):
        obs, input_, obs_mask =x
        if self.input_dim ==0:
            x1=obs
        else:
            x1=torch.cat([obs,input_],dim=-1)
        if self.obs_mask_enabled:
            x1=torch.cat([x1,obs_mask],dim=-1)

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
        if not self.discrete_state:
            m=self.out_mu(x)
            s=torch.clip(torch.nn.functional.softplus(self.out_sigma(x)),1.0e-10,1.0)
            m=torch.clip(m,-1.0,1.0)
            return torch.distributions.Normal(m,s)
        else:
            m=self.out_mu(x)
            return torch.distributions.RelaxedOneHotCategorical(logits=m,temperature=1)



