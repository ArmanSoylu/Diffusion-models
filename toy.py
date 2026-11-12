import os
import sys
from gc import freeze
import math

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import make_moons



T = 1000
NUM_CLASSES = 2
NULL_CLASS = NUM_CLASSES

P_UNCOND = 0.1

OUT = "out"


def make_data(n=20000,noise = 0.05):

    X0_np,y_np = make_moons(n_samples=n,noise=noise)

    X0 = torch.from_numpy(X0_np)
    y = torch.from_numpy(y_np)

    X0 = (X0 - X0.mean(dim=0)) / X0.std(dim=0)
    return X0,y

#NOISE CALCULATION



def make_schedule(num_steps=T, beta_start =1e-4,beta_end=0.02):

    betas = torch.linspace(beta_start,beta_end,num_steps,dtype=torch.float32)
    alphas = 1.0-betas
    alpha_bars = torch.cumprod(alphas,0)
    return betas,alphas,alpha_bars


#FORWARD PROCESS

def q_sample(x0,t,eps,alpha_bars):

    a_bar= alpha_bars[t]

    a_bar.unsqueeze_(-1)

    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1- a_bar) * eps
    return x_t


#MODEL

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2

        i = torch.arange( half_dim,dtype=torch.float32)

        freqs = torch.exp(-math.log(-10000.0)*i/half_dim-1)
        self.register_buffer('freqs',freqs)

    def forward(self,t):

        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)

        emb = torch.cat([args,t],dim=-1)

        return emb

class ToyModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, emb_dim=64, hidden=256):
        super().__init__()

        self.time = TimeEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim),nn.SiLU(),nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes, emb_dim)

        in_dim = 2+ emb_dim
        self.body = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.SiLU(),
    nn.Linear(hidden, hidden), nn.SiLU(),
    nn.Linear(hidden, hidden), nn.SiLU(),
    nn.Linear(hidden, 2))


    def forward(self, x,t,y):
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)

        cond = t_emb+y_emb

        h = torch.cat((cond,t_emb),dim=-1)

        return self.body(h)











