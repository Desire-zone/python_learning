# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from alphabet import AlphaBet
class BiLstm(nn.Module):
    def __init__(self, hyperpara):
        super(BiLstm, self).__init__()
        self.hyperpara = hyperpara
        self.wordAlpha = AlphaBet
        self.embed_num = self.hyperpara.embedding_num
        self.embed_dim = self.hyperpara.embedding_dim
        self.class_num = self.hyperpara.tag_size
        self.hidden_dim = self.hyperpara.LSTM_hidden_dim
        self.num_layers = self.hyperpara.num_layers
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        self.inputdropout = nn.Dropout(self.hyperpara.inputdropout)
        self.outputdropout = nn.Dropout(self.hyperpara.outputdropout)
        self.BiLstm = nn.LSTM(self.embed_dim, self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True)
        self.fc1 = nn.Linear(self.hidden_dim*2, self.class_num, bias=True)
        
    def init_hidden(self, num_layers, batch_size):
        return (Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim)),
                Variable(torch.zeros(num_layers * 2, batch_size, self.hidden_dim)))

    def forward(self, x):
        wordRepresents = self.embed(x) 
        wordRepresents = self.inputdropout(wordRepresents)
        BiLstmOutputs,_ = self.BiLstm(wordRepresents)
        self.outputdropout(BiLstmOutputs)
        dim = BiLstmOutputs.size(2)
        BiLstmOutputs1 = BiLstmOutputs.contiguous().view(-1, dim)     
        tagHiddens = self.fc1(BiLstmOutputs1)
        return tagHiddens    
