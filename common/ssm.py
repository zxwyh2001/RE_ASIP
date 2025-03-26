import torch
import torch.nn as nn
from modules import Mlp

class SPM(nn.Module):
    def __init__(self, in_joints=6, in_features=32):
        super().__init__()
        
        self.explict_space_prior = torch.load('./dataset/spatial_prior.pt')
        self.explict_space_prior = nn.Parameter(self.explict_space_prior,requires_grad=False)
        
        # self.explict_space_prior = nn.Parameter(torch.eye(6),requires_grad=False)
        
        self.implict_space_prior = nn.Parameter(torch.zeros((in_joints,in_joints)))
        
        self.space_gate = nn.Parameter(torch.zeros((in_joints,in_features)))
        
        self.norm = nn.LayerNorm(in_features)
        
        self.mlp = Mlp(in_features=in_features,hidden_features=in_features*2,out_features=in_features)

    def forward(self, x):
        # x : bs,f,j,d
        
        prior = self.explict_space_prior + self.implict_space_prior
        # prior = self.explict_space_prior
        x_ = x
        x_ = prior@x_
        x_ = self.norm(x_)
        x_ = self.mlp(x_)
        # x = x + x_
        return x_

class TPM(nn.Module):
    def __init__(self, frames=60, in_features=32):
        super().__init__()
        # self.explict_temporal_prior = torch.load('./dataset/temporal_prior_f'+str(frames)+'.pt')
        self.explict_temporal_prior = torch.load('./dataset/temporal_prior.pt')
        self.explict_temporal_prior = nn.Parameter(self.explict_temporal_prior, requires_grad=False)
        
        # self.explict_temporal_prior = nn.Parameter(torch.eye(frames), requires_grad=False)
        
        self.implict_temporal_prior = nn.Parameter(torch.zeros((frames,frames)))
        
        self.temporal_gate = nn.Parameter(torch.zeros((frames,in_features)))
        
        self.norm = nn.LayerNorm(in_features)
        
        self.mlp = Mlp(in_features=in_features,hidden_features=in_features*2,out_features=in_features)

    def forward(self, x):
        # x : bs,j,f,d
        # prior = self.explict_temporal_prior + self.implict_temporal_prior
        prior = self.explict_temporal_prior
        x_ = x
        x_ = prior@x_
        x_ = self.norm(x_)
        x_ = self.mlp(x_)
        # x = x + x_
        return x_