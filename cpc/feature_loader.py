# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torchaudio
import os
import json
import argparse
from .cpc_default_config import get_default_cpc_config
from .dataset import parseSeqLabels
from .model import CPCModel, ConcatenatedModel, CPCBertModel
import pickle

class FeatureModule(torch.nn.Module):
    r"""
    A simpler interface to handle CPC models. Useful for a smooth workflow when
    working with CPC trained features.
    """

    def __init__(self, featureMaker, get_encoded, collapse=False, cca_projection=None):
        super(FeatureModule, self).__init__()
        self.get_encoded = get_encoded
        self.featureMaker = featureMaker
        self.collapse = collapse
        self.cca_projection = None
        if cca_projection:
            assert cca_projection[-4:] == ".pkl"
            print("Loading canonical correlation analysis model.")
            with open(cca_projection, 'rb') as cca_file:
                self.cca_projection = pickle.load(cca_file)

    @property
    def out_feature_dim(self):
        if self.get_encoded:
            return self.featureMaker.gEncoder.getDimOutput()
        return self.featureMaker.gAR.getDimOutput()

    def getDownsamplingFactor(self):
        return self.featureMaker.gEncoder.DOWNSAMPLING

    def forward(self, data):
        batchAudio, label = data
        if len(batchAudio.size()) == 4:
            batchAudio = batchAudio[:, 0]
        cFeature, encoded, _ = self.featureMaker(batchAudio.cuda(), label)
        if self.get_encoded:
            cFeature = encoded
        if self.collapse:
            cFeature = cFeature.contiguous().view(-1, cFeature.size(2))
        if self.cca_projection:
            cFeature_np = self.cca_projection.transform(cFeature.cpu().numpy())
            cFeature = torch.tensor(cFeature_np, dtype=cFeature.dtype, device=cFeature.device)
        return cFeature


class CPCModule(torch.nn.Module):

    def __init__(self,
                 feature_maker,
                 cpc_criterion,
                 main_distance_only=False,
                 n_pred = -1):

        super(CPCModule, self).__init__()
        self.feature_maker = feature_maker
        self.cpc_criterion = cpc_criterion
        self.n_pred = n_pred
        self.main_distance_only = main_distance_only

    def getDownsamplingFactor(self):
        return self.feature_maker.gEncoder.DOWNSAMPLING

    def forward(self, data):
        batchAudio, label = data
        cFeature, encoded, label = self.feature_maker(batchAudio.cuda(), label)
        if self.main_distance_only:
            predictions = self.cpc_criterion.getCosineDistances(cFeature, encoded)[self.n_pred]
        else:
            predictions = self.cpc_criterion.getPrediction(cFeature, encoded, label)[0][self.n_pred]
            predictions = torch.nn.functional.softmax(predictions, dim=1)
        return predictions


class ModelPhoneCombined(torch.nn.Module):
    r"""
    Concatenates a CPC feature maker and a phone predictor.
    """

    def __init__(self, model, criterion, oneHot):
        r"""
        Arguments:
            model (FeatureModule): feature maker
            criterion (PhoneCriterion): phone predictor
            oneHot (bool): set to True to get a one hot output
        """
        super(ModelPhoneCombined, self).__init__()
        self.model = model
        self.criterion = criterion
        self.oneHot = oneHot

    def getDownsamplingFactor(self):
        return self.model.getDownsamplingFactor()

    def forward(self, data):
        c_feature = self.model(data)
        pred = self.criterion.getPrediction(c_feature)
        P = pred.size(2)

        if self.oneHot:
            pred = pred.argmax(dim=2)
            pred = toOneHot(pred, P)
        else:
            pred = torch.nn.functional.softmax(pred, dim=2)
        return pred


class ModelClusterCombined(torch.nn.Module):
    r"""
    Concatenates a CPC feature maker and a clustering module.
    """

    def __init__(self, model, cluster, nk, outFormat):

        if outFormat not in ['oneHot', 'int', 'softmax']:
            raise ValueError(f'Invalid output format {outFormat}')

        super(ModelClusterCombined, self).__init__()
        self.model = model
        self.cluster = cluster
        self.nk = nk
        self.outFormat = outFormat

    def getDownsamplingFactor(self):
        return self.model.getDownsamplingFactor()

    def forward(self, data):
        c_feature = self.model(data)
        pred = self.cluster(c_feature)
        if self.outFormat == 'oneHot':
            pred = pred.min(dim=2)[1]
            pred = toOneHot(pred, self.nk)
        elif self.outFormat == 'int':
            pred = pred.min(dim=2)[1]
        else:
            pred = torch.nn.functional.softmax(-pred, dim=2)
        return pred


def loadArgs(args, locArgs, forbiddenAttr=None):
    for k, v in vars(locArgs).items():
        if forbiddenAttr is not None:
            if k not in forbiddenAttr:
                setattr(args, k, v)
        else:
            setattr(args, k, v)


def loadSupervisedCriterion(pathCheckpoint):
    from cpc.criterion import CTCPhoneCriterion, PhoneCriterion

    *_, args = getCheckpointData(os.path.dirname(pathCheckpoint))
    _, nPhones = parseSeqLabels(args.pathPhone)
    if False:#args.CTC:
        criterion = CTCPhoneCriterion(args.hiddenGar if not args.onEncoder
                                      else args.hiddenEncoder,
                                      nPhones, args.onEncoder)
    else:
        criterion = PhoneCriterion(args.hiddenGar, nPhones, args.onEncoder)

    state_dict = torch.load(pathCheckpoint)
    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion, nPhones


def getCheckpointData(pathDir):

    if not os.path.isdir(pathDir):
        return None
    checkpoints = [x for x in os.listdir(pathDir)
                   if os.path.splitext(x)[1] == '.pt'
                   and os.path.splitext(x[11:])[0].isdigit()]
    if len(checkpoints) == 0:
        print("No checkpoints found at " + pathDir)
        return None
    checkpoints.sort(key=lambda x: int(os.path.splitext(x[11:])[0]))
    data = os.path.join(pathDir, checkpoints[-1])

    with open(os.path.join(pathDir, 'checkpoint_logs.json'), 'rb') as file:
        logs = json.load(file)

    with open(os.path.join(pathDir, 'checkpoint_args.json'), 'rb') as file:
        args = json.load(file)

    args = argparse.Namespace(**args)
    defaultArgs = get_default_cpc_config()
    loadArgs(defaultArgs, args)

    return os.path.abspath(data), logs, defaultArgs


def getEncoder(args):

    if args.encoder_type == 'mfcc':
        from .model import MFCCEncoder
        return MFCCEncoder(args.hiddenEncoder)
    elif args.encoder_type == 'lfb':
        from .model import LFBEnconder
        return LFBEnconder(args.hiddenEncoder)
    else:
        from .model import CPCEncoder
        return CPCEncoder(args.hiddenEncoder, args.normMode)


def getAR(args):
    if args.arMode == 'transformer':
        from .transformers import buildTransformerAR
        arNet = buildTransformerAR(args.hiddenEncoder, args.hiddenGar, args.nLevelsGRU,
                                   args.sizeWindow // 160, args.abspos)
        args.hiddenGar = args.hiddenEncoder
    elif args.cpc_mode == "bert":
        from .model import BiDIRARTangled
        arNet = BiDIRARTangled(args.hiddenEncoder, args.hiddenGar,
                               args.nLevelsGRU)
    elif args.arMode == 'no_ar':
        from .model import NoAr
        arNet = NoAr()
    else:
        from .model import CPCAR
        arNet = CPCAR(args.hiddenEncoder, args.hiddenGar,
                      args.samplingType == "sequential",
                      args.nLevelsGRU,
                      mode=args.arMode,
                      reverse=args.cpc_mode == "reverse")
    return arNet


def loadModel(pathCheckpoints, loadStateDict=True, updateConfig=None):
    models = []
    hiddenGar, hiddenEncoder = 0, 0
    for path in pathCheckpoints:
        print(f"Loading checkpoint {path}")
        _, _, locArgs = getCheckpointData(os.path.dirname(path))
        doLoad = locArgs.load is not None and \
            (len(locArgs.load) > 1 or
             os.path.dirname(locArgs.load[0]) != os.path.dirname(path))

        if updateConfig is not None and not doLoad:
            print(f"Updating the configuration file with ")
            print(f'{json.dumps(vars(updateConfig), indent=4, sort_keys=True)}')
            loadArgs(locArgs, updateConfig)

        if doLoad:
            m_, hg, he = loadModel(locArgs.load,
                                   loadStateDict=False,
                                   updateConfig=updateConfig)
            hiddenGar += hg
            hiddenEncoder += he
        else:
            encoderNet = getEncoder(locArgs)

            arNet = getAR(locArgs)
            if locArgs.cpc_mode == "bert":
                m_ = CPCBertModel(encoderNet, arNet,
                                  blockSize=locArgs.nPredicts)
                m_.supervised = locArgs.supervised
            else:
                m_ = CPCModel(encoderNet, arNet)

        if loadStateDict:
            print(f"Loading the state dict at {path}")
            state_dict = torch.load(path, 'cpu')
            m_.load_state_dict(state_dict["gEncoder"], strict=False)
        if not doLoad:
            hiddenGar += locArgs.hiddenGar
            hiddenEncoder += locArgs.hiddenEncoder

        models.append(m_)

    if len(models) == 1:
        return models[0], hiddenGar, hiddenEncoder

    return ConcatenatedModel(models), hiddenGar, hiddenEncoder


def get_module(i_module):
    if isinstance(i_module, torch.nn.DataParallel):
        return get_module(i_module.module)
    if isinstance(i_module, FeatureModule):
        return get_module(i_module.featureMaker)
    if isinstance(i_module, torch.nn.parallel.DistributedDataParallel):
        return get_module(i_module.module)
    return i_module


def save_checkpoint(model_state, criterion_state, optimizer_state, best_state,
                    path_checkpoint):

    state_dict = {"gEncoder": model_state,
                  "cpcCriterion": criterion_state,
                  "optimizer": optimizer_state,
                  "best": best_state}

    torch.save(state_dict, path_checkpoint)


def toOneHot(inputVector, nItems):

    batchSize, seqSize = inputVector.size()
    out = torch.zeros((batchSize, seqSize, nItems),
                      device=inputVector.device, dtype=torch.long)
    out.scatter_(2, inputVector.view(batchSize, seqSize, 1), 1)
    return out


def seqNormalization(out):
    # out.size() = Batch x Seq x Channels
    mean = out.mean(dim=1, keepdim=True)
    var = out.var(dim=1, keepdim=True)
    return (out - mean) / torch.sqrt(var + 1e-08)


def buildFeature(featureMaker, seqPath, strict=False,
                 maxSizeSeq=64000, seqNorm=False):
    r"""
    Apply the featureMaker to the given file.
    Arguments:
        - featureMaker (FeatureModule): model to apply
        - seqPath (string): path of the sequence to load
        - strict (bool): if True, always work with chunks of the size
                         maxSizeSeq
        - maxSizeSeq (int): maximal size of a chunk
        - seqNorm (bool): if True, normalize the output along the time
                          dimension to get chunks of mean zero and var 1
    Return:
        a torch vector of size 1 x Seq_size x Feature_dim
    """
    seq = torchaudio.load(seqPath)[0]
    sizeSeq = seq.size(1)
    start = 0
    out = []
    while start < sizeSeq:
        if strict and start + maxSizeSeq > sizeSeq:
            break
        end = min(sizeSeq, start + maxSizeSeq)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = featureMaker((subseq, None))
            if seqNorm:
                features = seqNormalization(features)

        out.append(features.detach().cpu())
        start += maxSizeSeq

    if strict and start < sizeSeq:
        subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = featureMaker((subseq, None))
            if seqNorm:
                features = seqNormalization(features)

        delta = (sizeSeq - start) // featureMaker.getDownsamplingFactor()
        out.append(features[:, -delta:].detach().cpu())

    out = torch.cat(out, dim=1)

    return out


def buildFeature_batch(featureMaker, seqPath, strict=False,
                       maxSizeSeq=8000, seqNorm=False, batch_size=8):
    r"""
    Apply the featureMaker to the given file. Apply batch-computation
    Arguments:
        - featureMaker (FeatureModule): model to apply
        - seqPath (string): path of the sequence to load
        - strict (bool): if True, always work with chunks of the size
                         maxSizeSeq
        - maxSizeSeq (int): maximal size of a chunk
        - seqNorm (bool): if True, normalize the output along the time
                          dimension to get chunks of mean zero and var 1
    Return:
        a torch vector of size 1 x Seq_size x Feature_dim
    """
    if next(featureMaker.parameters()).is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    seq = torchaudio.load(seqPath)[0]
    sizeSeq = seq.size(1)

    # Compute number of batches
    n_chunks = sizeSeq // maxSizeSeq
    n_batches = n_chunks // batch_size
    if n_chunks % batch_size != 0:
        n_batches += 1

    out = []
    # Treat each batch
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size * maxSizeSeq
        end = min((batch_idx + 1) * batch_size * maxSizeSeq, maxSizeSeq * n_chunks)
        batch_seqs = (seq[:, start:end]).view(-1, 1, maxSizeSeq).to(device)
        with torch.no_grad():
            # breakpoint()
            batch_out = featureMaker((batch_seqs, None))
            for features in batch_out:
                features = features.unsqueeze(0)
                if seqNorm:
                    features = seqNormalization(features)
                out.append(features.detach().cpu())

    # Remaining frames
    if sizeSeq % maxSizeSeq >= featureMaker.getDownsamplingFactor():
        remainders = sizeSeq % maxSizeSeq
        if strict:
            subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1).to(device)
            with torch.no_grad():
                features = featureMaker((subseq, None))
                if seqNorm:
                    features = seqNormalization(features)
            delta = remainders // featureMaker.getDownsamplingFactor()
            out.append(features[:, -delta:].detach().cpu())
        else:
            subseq = (seq[:, -remainders:]).view(1, 1, -1).to(device)
            with torch.no_grad():
                features = featureMaker((subseq, None))
                if seqNorm:
                    features = seqNormalization(features)
            out.append(features.detach().cpu())

    out = torch.cat(out, dim=1)
    return out
