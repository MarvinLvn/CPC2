# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 sizeSeq,         # Size of the input sequence
                 dk,              # Dimension of the input sequence
                 dropout,         # Dropout parameter
                 relpos=False):   # Do we retrieve positional information ?
        super(ScaledDotProductAttention, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.relpos = relpos
        self.sizeSeq = sizeSeq

        if relpos:
            self.Krelpos = nn.Parameter(torch.Tensor(dk, sizeSeq))
            self.initmat_(self.Krelpos)
            self.register_buffer('z', torch.zeros(1, sizeSeq, 1))

        # A mask is set so that a node never queries data in the future
        mask = torch.tril(torch.ones(sizeSeq, sizeSeq), diagonal=0)
        mask = 1 - mask
        mask[mask == 1] = -float('inf')
        self.register_buffer('mask', mask.unsqueeze(0))

    def initmat_(self, mat, dim=0):
        stdv = 1. / math.sqrt(mat.size(dim))
        mat.data.uniform_(-stdv, stdv)

    def prepare(self, x):
        # Input dim : N x S x dk
        N, S, k = x.size()

        r_ = S % self.sizeSeq
        if r_ > 0:
            to_add = torch.zeros(size=(N, self.sizeSeq -r_, k),
                                 device=x.device,
                                 dtype=x.dtype)
            x = torch.cat([x, to_add], dim=1)
            S += self.sizeSeq -r_

        return x.view(N * (S // self.sizeSeq), self.sizeSeq, k)

    def forward(self, Q, K, V):
        # Input dim : N x sizeSeq x dk
        N, S, k = Q.size()
        Q = self.prepare(Q)
        K = self.prepare(K)
        V = self.prepare(V)

        QK = torch.bmm(Q, K.transpose(-2, -1))

        if self.relpos:
            bsz = Q.size(0)
            QP = Q.matmul(self.Krelpos)
            # This trick with z fills QP's diagonal with zeros
            QP = torch.cat((self.z.expand(bsz, -1, -1), QP), 2)
            QK += QP.view(bsz, self.sizeSeq + 1, self.sizeSeq)[:, 1:, :]
        A = self.softmax(QK / math.sqrt(K.size(-1)) + self.mask)
        out = torch.bmm(self.drop(A), V)
        out = out.view(N, -1, k)[:, :S]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 sizeSeq,   # Size of a sequence
                 dropout,   # Dropout parameter
                 dmodel,    # Model's dimension
                 nheads,    # Number of heads in the model
                 abspos):   # Is positional information encoded in the input ?
        super(MultiHeadAttention, self).__init__()
        self.Wo = nn.Linear(dmodel, dmodel, bias=False)
        self.Wk = nn.Linear(dmodel, dmodel, bias=False)
        self.Wq = nn.Linear(dmodel, dmodel, bias=False)
        self.Wv = nn.Linear(dmodel, dmodel, bias=False)
        self.nheads = nheads
        self.dk = dmodel // nheads
        self.Att = ScaledDotProductAttention(sizeSeq, self.dk,
                                             dropout, not abspos)

    def trans_(self, x):
        bsz, bptt, h, dk = x.size(0), x.size(1), self.nheads, self.dk
        return x.view(bsz, bptt, h, dk).transpose(1, 2).contiguous().view(bsz * h, bptt, dk)

    def reverse_trans_(self, x):
        bsz, bptt, h, dk = x.size(
            0) // self.nheads, x.size(1), self.nheads, self.dk
        return x.view(bsz, h, bptt, dk).transpose(1, 2).contiguous().view(bsz, bptt, h * dk)

    def forward(self, Q, K, V):
        q = self.trans_(self.Wq(Q))
        k = self.trans_(self.Wk(K))
        v = self.trans_(self.Wv(V))
        y = self.reverse_trans_(self.Att(q, k, v))
        return self.Wo(y)


class FFNetwork(nn.Module):
    def __init__(self, din, dout, dff, dropout):
        super(FFNetwork, self).__init__()
        self.lin1 = nn.Linear(din, dff, bias=True)
        self.lin2 = nn.Linear(dff, dout, bias=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(self.relu(self.lin1(x))))


class TransformerLayer(nn.Module):
    def __init__(self, sizeSeq=32, dmodel=512, dout=512, dff=2048,
                 dropout=0.1, nheads=8,
                 abspos=False):
        super(TransformerLayer, self).__init__()
        self.multihead = MultiHeadAttention(sizeSeq, dropout,
                                            dmodel, nheads, abspos)
        self.ln_multihead = nn.LayerNorm(dmodel)
        self.ffnetwork = FFNetwork(dmodel, dmodel, dff, dropout)
        # last linear layer to reduce dimension
        self.last_linear = nn.Linear(dmodel, dout)
        self.ln_ffnetwork = nn.LayerNorm(dout)

    def forward(self, x):
        y = self.ln_multihead(x + self.multihead(Q=x, K=x, V=x))
        return self.ln_ffnetwork(self.last_linear(y + self.ffnetwork(y)))


class MultiClassifierTransformerHead(nn.Module):

    def __init__(self, nclassifiers, sizeSeq=32, dmodel=512, dout=512, dff=2048,
                 dropout=0.1, nheads=8,
                 abspos=False):
        super(MultiClassifierTransformerHead, self).__init__()
        self.multihead = MultiHeadAttention(sizeSeq, dropout,
                                            dmodel, nheads, abspos)
        self.ln_multihead = nn.LayerNorm(dmodel)
        self.ffnetwork = FFNetwork(dmodel, dmodel * nclassifiers,dff,  dropout)
        # last linear layer to reduce dimension
        self.last_linear = nn.Linear(dmodel, dout)
        self.ln_ffnetwork = nn.LayerNorm(dout)
        self.nclassifiers = nclassifiers
        self.dout = dmodel

    def forward(self, x):
        y = self.ln_multihead(x + self.multihead(Q=x, K=x, V=x))
        B, S, _ = y.size()
        x = self.ffnetwork(y)
        y = y.view(B, S, 1, self.dout).expand(B, S, self.nclassifiers, self.dout)
        return self.ln_ffnetwork(self.last_linear(x.view(B, S, self.nclassifiers, self.dout) + y))


class StaticPositionEmbedding(nn.Module):
    def __init__(self, seqlen, dmodel):
        super(StaticPositionEmbedding, self).__init__()
        pos = torch.arange(0., seqlen).unsqueeze(1).repeat(1, dmodel)
        dim = torch.arange(0., dmodel).unsqueeze(0).repeat(seqlen, 1)
        div = torch.exp(- math.log(10000) * (2*(dim//2)/dmodel))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer('pe', pos.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def buildTransformerAR(dimEncoded,    # Output dimension of the encoder
                       dimAR,         # Output dimension of the auto-regressive network
                       nLayers,       # Number of transformer layers
                       sizeSeq,       # Expected size of the input sequence
                       abspos):
    layerSequence = []
    if abspos:
        layerSequence += [StaticPositionEmbedding(sizeSeq, dimAR)]
    layerSequence += [TransformerLayer(sizeSeq=sizeSeq,
                                       dmodel=dimAR, dout=dimEncoded, abspos=abspos)
                      for i in range(nLayers)]
    return nn.Sequential(*layerSequence)


def buildMultHeadTransformerAR(dimEncoded,    # Output dimension of the encoder
                               dimAR,         # Output dimension of the auto-regressive network
                               nLayers,       # Number of transformer layers
                               sizeSeq,       # Expected size of the input sequence
                               abspos,
                               nHeads):

    layerSequence = []
    if abspos:
        layerSequence += [StaticPositionEmbedding(sizeSeq, dimAR)]

    layerSequence += [TransformerLayer(sizeSeq=sizeSeq,
                                       dmodel=dimAR,
                                       dout=dimEncoded,
                                       abspos=abspos)
                      for i in range(nLayers - 1)]
    layerSequence += [MultiClassifierTransformerHead(nHeads,
                                                    dmodel=dimAR,
                                                    dout=dimEncoded,
                                                    sizeSeq=sizeSeq,
                                                    abspos=abspos)]

    return nn.Sequential(*layerSequence)
