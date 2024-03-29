# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple
import warnings
import torch

###########################################
# Networks
###########################################


class IDModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm"):

        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160


    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class MFCCEncoder(nn.Module):

    def __init__(self,
                 dimEncoded):

        super(MFCCEncoder, self).__init__()
        melkwargs = {"n_mels": max(128, dimEncoded), "n_fft": 321}
        self.dimEncoded = dimEncoded
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=dimEncoded,
                                               melkwargs=melkwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MFCC(x)
        return x.permute(0, 2, 1)


class LFBEnconder(nn.Module):

    def __init__(self, dimEncoded, normalize=True):

        super(LFBEnconder, self).__init__()
        self.dimEncoded = dimEncoded
        self.conv = nn.Conv1d(1, 2 * dimEncoded,
                              400, stride=1)
        self.register_buffer('han', torch.hann_window(400).view(1, 1, 400))
        self.instancenorm = nn.InstanceNorm1d(dimEncoded, momentum=1) \
            if normalize else None

    def forward(self, x):

        N, C, L = x.size()
        x = self.conv(x)
        x = x.view(N, self.dimEncoded, 2, -1)
        x = x[:, :, 0, :]**2 + x[:, :, 1, :]**2
        x = x.view(N * self.dimEncoded, 1,  -1)
        x = torch.nn.functional.conv1d(x, self.han, bias=None,
                                       stride=160, padding=350)
        x = x.view(N, self.dimEncoded,  -1)
        x = torch.log(1 + torch.abs(x))

        # Normalization
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                                   num_layers=nLevelsGRU, batch_first=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)
        else:
            self.baseNet = nn.GRU(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x


class NoAr(nn.Module):

    def __init__(self, *args):
        super(NoAr, self).__init__()

    def forward(self, x):
        return x


class BiDIRARTangled(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRARTangled, self).__init__()
        assert(dimOutput % 2 == 0)

        self.ARNet = nn.GRU(dimEncoded, dimOutput // 2,
                            num_layers=nLevelsGRU, batch_first=True,
                            bidirectional=True)

    def getDimOutput(self):
        return self.ARNet.hidden_size * 2

    def forward(self, x):

        self.ARNet.flatten_parameters()
        xf, _ = self.ARNet(x)
        return xf


class BiDIRAR(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRAR, self).__init__()
        assert(dimOutput % 2 == 0)

        self.netForward = nn.GRU(dimEncoded, dimOutput // 2,
                                 num_layers=nLevelsGRU, batch_first=True)
        self.netBackward = nn.GRU(dimEncoded, dimOutput // 2,
                                  num_layers=nLevelsGRU, batch_first=True)

    def getDimOutput(self):
        return self.netForward.hidden_size * 2

    def forward(self, x):

        self.netForward.flatten_parameters()
        self.netBackward.flatten_parameters()
        xf, _ = self.netForward(x)
        xb, _ = self.netBackward(torch.flip(x, [1]))
        return torch.cat([xf, torch.flip(xb, [1])], dim=2)


###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 mask_prob=0.0,
                 mask_length=10):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        self.mask_prob = mask_prob
        self.mask_length = mask_length

        # This would make more sense if the encoded features would be between 0 and 1.
        # Should think about normalizing
        if mask_prob > 0.0:
            self.mask_emb = nn.Parameter(
                torch.FloatTensor(encoder.dimEncoded).uniform_()
            )

    def compute_mask_indices(self,
            shape: Tuple[int, int],
            mask_prob: float,
            mask_length: int,
            min_masks: int = 0,
    ) -> np.ndarray:
        """
        Simplified version of the code that has been implemented for wav2vec 2.0:
        https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md

        Computes random mask spans for a given shape
        Args:
            shape: the shape for which to compute masks.
                should be of size 2 where first element is batch size and 2nd is timesteps
            mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
                number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
                however due to overlaps, the actual number will be smaller
            mask_length: length of a mask
            min_masks: minimum number of masked spans
        """
        bsz, all_sz = shape
        mask = np.full((bsz, all_sz), False)

        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * 100 * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

        mask_idcs = []
        for i in range(bsz):

            sz = all_sz
            num_mask = all_num_mask

            lengths = np.full(num_mask, mask_length)

            if sum(lengths) == 0:
                lengths[0] = min(mask_length, sz - 1)

            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

            mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

        min_len = min([len(m) for m in mask_idcs])
        nb_masked = 0
        for i, mask_idc in enumerate(mask_idcs):
            if len(mask_idc) > min_len:
                mask_idc = np.random.choice(mask_idc, min_len, replace=False)

            mask[i, mask_idc] = True
            nb_masked += len(mask_idc)

        percentage_masked = nb_masked / (bsz * all_sz)
        if percentage_masked > 0.6:
            warnings.warn("We detected that %.2f of all encoded frames have been masked. This might be too much." % percentage_masked)
        return mask

    def getMask(self, features):
        batchSize, seqSize, c = features.shape
        mask_indices = self.compute_mask_indices((batchSize, seqSize),
                                                 self.mask_prob,
                                                 self.mask_length,
                                                 min_masks=2)
        mask_indices = torch.from_numpy(mask_indices).to(features.device)
        features[mask_indices] = self.mask_emb
        return features

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)

        if self.mask_prob > 0.0:
            masked_encodedData = self.getMask(encodedData)
            cFeature = self.gAR(masked_encodedData)
        else:
            cFeature = self.gAR(encodedData)

        return cFeature, encodedData, label


class CPCBertModel(CPCModel):

    def __init__(self,
                 encoder,
                 AR,
                 nMaskSentence=2,
                 blockSize=12):

        super(CPCBertModel, self).__init__(encoder, AR)
        self.blockSize = blockSize
        self.nMaskSentence = nMaskSentence
        self.supervised = False

    def getMask(self, batchData):

        batchSize, seqSize, c = batchData.size()
        maskLabel = torch.randint(0, seqSize // self.blockSize,
                                  (self.nMaskSentence * batchSize, 1))
        maskLabel *= self.blockSize

        baseX = torch.arange(0, self.blockSize, dtype=torch.long)
        baseX = baseX.expand(self.nMaskSentence * batchSize, self.blockSize)
        maskLabel = maskLabel + baseX
        maskLabel = maskLabel.view(-1)

        baseY = torch.arange(0, batchSize,
                             dtype=torch.long).view(-1, 1) * seqSize
        baseY = baseY.expand(batchSize,
                             self.nMaskSentence *
                             self.blockSize).contiguous().view(-1)
        maskLabel = maskLabel + baseY
        outLabels = torch.zeros(batchSize * seqSize,
                                dtype=torch.uint8)
        outLabels[maskLabel] = 1

        outLabels = outLabels.view(batchSize, seqSize)

        return outLabels

    def forward(self, batchData, label):

        fullEncoded = self.gEncoder(batchData).permute(0, 2, 1)

        # Sample random blocks of data
        if not self.supervised:
            maskLabels = self.getMask(fullEncoded)
            partialEncoded = fullEncoded.clone()
            partialEncoded[maskLabels] = 0
            cFeature = self.gAR(partialEncoded)
            return cFeature, fullEncoded, maskLabels.cuda()

        else:
            cFeature = self.gAR(fullEncoded)
            return cFeature, fullEncoded, label


class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):

        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):

        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData, label = model(batchData, label)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), \
            torch.cat(outEncoded, dim=2), label
