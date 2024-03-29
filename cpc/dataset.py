# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import os
import random
import time
from copy import deepcopy
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import functools

import tqdm
from torch.multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

class AudioBatchData(Dataset):

    def __init__(self,
                 path,
                 sizeWindow,
                 seqNames,
                 phoneLabelsDict,
                 nSpeakers,
                 nProcessLoader=10,
                 MAX_SIZE_LOADED=4000000000,
                 transform=None,
                 augment_past=False,
                 augment_future=False,
                 augmentation=None,
                 keep_temporality=True,
                 past_equal_future=False,
                 signal_quality_path=None,
                 signal_quality_step=1600,
                 signal_quality_mode=None):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabelsDict (dictionnary): if not None, a dictionnary with the
                                             following entries

                                             "step": size of a labelled window
                                             "$SEQ_NAME": list of phoneme labels for
                                             the sequence $SEQ_NAME
           - nSpeakers (int): number of speakers to expect.
           - nProcessLoader (int): number of processes to call when loading the
                                   data from the disk
           - MAX_SIZE_LOADED (int): target maximal size of the floating array
                                    containing all loaded data.
        """
        # Data parameters
        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.nProcessLoader = nProcessLoader
        self.dbPath = Path(path)
        self.sizeWindow = sizeWindow
        self.seqNames = [(s, self.dbPath / x) for s, x in seqNames]
        self.reload_pool = Pool(nProcessLoader)
        self.transform = transform
        self.keep_temporality = keep_temporality

        # Signal quality parameters
        self.signal_quality_path = Path(signal_quality_path) if signal_quality_path is not None else None
        self.signal_quality_step = signal_quality_step
            # Number of signal quality estimations in one input sequence of 1.28 sec
        self.signal_quality_size = self.sizeWindow // self.signal_quality_step
        self.signal_quality_mode = signal_quality_mode

        if self.signal_quality_path is not None:
            self.init_min_max_signal_quality()


        # Data augmentation parameters
        self.augment_past = augment_past
        self.augment_future = augment_future
        self.augmentation = augmentation
        self.past_equal_future = past_equal_future
        if self.past_equal_future and not self.augment_past:
            raise ValueError("Can only apply the same transformation on past and future sequences,"
                             "when past sequence is augmented. Here --augment_past = False")

        # Initialize data loading
        self.doubleLabels = False

        self.prepare()
        self.speakers = list(range(nSpeakers))
        self.data = []
        self.data_quality = []

        self.phoneSize = 0 if phoneLabelsDict is None else \
            phoneLabelsDict["step"]
        self.phoneStep = 0 if phoneLabelsDict is None else \
            self.sizeWindow // self.phoneSize

        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.loadNextPack(first=True)
        self.loadNextPack()

    def init_min_max_signal_quality(self):
        file_path = self.signal_quality_path / 'min_max.csv'
        if not file_path.is_file():
            raise FileNotFoundError('Can not find file containing min/max values of snr and c50 under: %s' % (
                    self.signal_quality_path / 'min_max.csv'))
        with open(file_path, 'r') as fin:
            reader = csv.reader(fin)
            keys = next(reader)
            values = next(reader)
            data = {k: v for k, v in zip(keys, values)}
            try:
                self.min_snr, self.max_snr = float(data['min_snr']), float(data['max_snr'])
                self.min_c50, self.max_c50 = float(data['min_c50']), float(data['max_c50'])
            except:
                raise ValueError("min_max.csv should contain the following keys: min_snr, max_snr, min_c50, max_c50.")


    def resetPhoneLabels(self, newPhoneLabels, step):
        self.phoneSize = step
        self.phoneStep = self.sizeWindow // self.phoneSize
        self.phoneLabelsDict = deepcopy(newPhoneLabels)
        self.loadNextPack()

    def splitSeqTags(seqName):
        path = os.path.normpath(seqName)
        return path.split(os.sep)

    def getSeqNames(self):
        return [str(x[1]) for x in self.seqNames]

    def clear(self):
        if 'data' in self.__dict__:
            del self.data
        if 'speakerLabel' in self.__dict__:
            del self.speakerLabel
        if 'phoneLabels' in self.__dict__:
            del self.phoneLabels
        if 'seqLabel' in self.__dict__:
            del self.seqLabel
        if 'dataSpkr' in self.__dict__:
            del self.dataSpkr

    def prepare(self):
        if self.keep_temporality:
            # We shuffle sequences but keeping set of sequences that happen in the same session
            seqNames_by_blocks = []
            curr_seq_id = None
            for seq_id, seq_path in self.seqNames:
                if curr_seq_id != seq_id:
                    seqNames_by_blocks.append([(seq_id, seq_path)])
                    curr_seq_id = seq_id
                else:
                    seqNames_by_blocks[-1].append((seq_id, seq_path))
            random.shuffle(seqNames_by_blocks)
            self.seqNames = [item for sublist in seqNames_by_blocks for item in sublist]
        else:
            # We shuffle sequences in random order
            random.shuffle(self.seqNames)

        # Let's not forget to keep seqNames and signal_quality_names aligned
        if self.signal_quality_path is not None:
            self.signal_quality_names = [self.signal_quality_path / os.path.relpath(x, self.dbPath).replace('.wav', '.pt')
                                         for s, x in self.seqNames]
        start_time = time.time()

        print("Checking length...")
        allLength = self.reload_pool.map(extractLength, self.seqNames)

        self.packageIndex, self.totSize = [], 0
        start, packageSize = 0, 0
        for index, length in tqdm.tqdm(enumerate(allLength)):
            packageSize += length
            if packageSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0

        if packageSize > 0:
            self.packageIndex.append([start, len(self.seqNames)])
            self.totSize += packageSize

        print(f"Done, elapsed: {time.time() - start_time:.3f} seconds")
        print(f'Scanned {len(self.seqNames)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0

    def getNPacks(self):
        return len(self.packageIndex)

    def loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            start_time = time.time()
            print('Joining pool')
            self.r.wait()
            print(f'Joined process, elapsed={time.time()-start_time:.3f} secs')
            self.nextData = self.r.get()
            self.parseNextDataBlock()
            del self.nextData

        self.nextPack = (self.currentPack + 1) % len(self.packageIndex)
        seqStart, seqEnd = self.packageIndex[self.nextPack]

        if self.nextPack == 0 and len(self.packageIndex) > 1:
            self.prepare()

        if self.signal_quality_path is not None:
            partial_loadFile = functools.partial(loadFile, signal_quality_step=self.signal_quality_step)
            self.r = self.reload_pool.map_async(partial_loadFile, zip(self.seqNames[seqStart:seqEnd],
                                                                      self.signal_quality_names[seqStart:seqEnd]))
        else:
            self.r = self.reload_pool.map_async(loadFile, self.seqNames[seqStart:seqEnd])

    def parseNextDataBlock(self):
        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]
        self.phoneLabels = []
        speakerSize = 0
        indexSpeaker = 0

        # To accelerate the process a bit
        self.nextData.sort(key=lambda x: (x[0], x[1]))
        tmpData = []
        tmpDataQuality = []

        for speaker, seqName, seq, *signal_quality in self.nextData:
            while self.speakers[indexSpeaker] < speaker:
                indexSpeaker += 1
                self.speakerLabel.append(speakerSize)
            if self.speakers[indexSpeaker] != speaker:
                raise ValueError(f'{speaker} invalid speaker')

            if self.phoneLabelsDict is not None:
                self.phoneLabels += self.phoneLabelsDict[seqName]
                newSize = len(self.phoneLabelsDict[seqName]) * self.phoneSize
                seq = seq[:newSize]
            sizeSeq = seq.size(0)
            tmpData.append(seq)
            if len(signal_quality) != 0:
                tmpDataQuality.append(signal_quality[0])
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq
            del seq
            del signal_quality

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(tmpData, dim=0)
        if len(tmpDataQuality) != 0:
            self.data_quality = torch.cat(tmpDataQuality, dim=0)
            # Normalize scores
            self.data_quality[:, 0] = (self.data_quality[:, 0] - self.min_snr) / (self.max_snr - self.min_snr)
            self.data_quality[:, 1] = (self.data_quality[:, 1] - self.min_c50) / (self.max_c50 - self.min_c50)

            # Add average of snr and C50 as a column
            self.data_quality = torch.cat((self.data_quality,
                                           torch.mean(self.data_quality, dim=1).view(-1, 1)), dim=1)

    def getPhonem(self, idx):
        idPhone = idx // self.phoneSize
        return self.phoneLabels[idPhone:(idPhone + self.phoneStep)]

    def getSignalQuality(self, idx):
        idxQuality = idx // self.signal_quality_step
        estimated_signal_quality = self.data_quality[idxQuality:(idxQuality + self.signal_quality_size)]
        if self.signal_quality_mode == 'snr':
            return estimated_signal_quality[:, 0]
        elif self.signal_quality_mode == 'c50':
            return estimated_signal_quality[:, 1]
        elif self.signal_quality_mode == 'snr_c50':
            return estimated_signal_quality[:, 2]
        else:
            raise ValueError("--signal_quality_mode should be in ['snr', 'c50', 'snr_c50'].")

    def getSpeakerLabel(self, idx):
        idSpeaker = next(x[0] for x in enumerate(
            self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker

    def __len__(self):
        return self.totSize // self.sizeWindow

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)
            print("upper bound %d" % (len(self.data) - self.sizeWindow - 1))

        outData = self.data[idx:(self.sizeWindow + idx)].view(1, -1)
        label = torch.tensor(self.getSpeakerLabel(idx), dtype=torch.long)
        if self.phoneSize > 0:
            label_phone = torch.tensor(self.getPhonem(idx), dtype=torch.long)
            if not self.doubleLabels:
                label = label_phone
        else:
            label_phone = torch.zeros(1)

        if self.transform is not None:
            outData = self.transform(outData)

        x1, x2 = outData, outData
        # Past augmentation only
        if self.augment_past and self.augmentation:
            x1 = self.augmentation(x1)

        # And/or future augmentation
        if not self.past_equal_future and self.augment_future and self.augmentation:
            x2 = self.augmentation(x2)
        # If past_equal_future is activated, apply the same transformation to past and future
        if self.past_equal_future:
            x2 = x1

        x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
        outData = torch.cat([x1, x2], dim=0)

        res = outData, label
        if self.doubleLabels:
            res = res + (label_phone, )

        if self.signal_quality_path:
            outDataSignalQuality = self.getSignalQuality(idx)
            res = res + (outDataSignalQuality,)
        return res

    def getNSpeakers(self):
        return len(self.speakers)

    def getNSeqs(self):
        return len(self.seqLabel) - 1

    def getNLoadsPerEpoch(self):
        return len(self.packageIndex)

    def getBaseSampler(self, type, batchSize,
                       offset, batchSizePerGPU=None):

        if type == "samespeaker":
            return SameSpeakerSampler(batchSize, self.speakerLabel,
                                      self.sizeWindow, offset)
        if type == "samesequence":
            return SameSpeakerSampler(batchSize, self.seqLabel,
                                      self.sizeWindow, offset)
        if type == "temporalsamespeaker":
            return TemporalSameSpeakerSampler(batchSize,
                                              self.speakerLabel,
                                              self.sizeWindow, offset,
                                              batchSizePerGPU=batchSizePerGPU)
        if type == "sequential":
            return SequentialSampler(len(self.data), self.sizeWindow,
                                     offset, batchSize)

        if type == "uniform":
            sampler = UniformAudioSampler(len(self.data), self.sizeWindow,
                                          offset)
            return BatchSampler(sampler, batchSize, True)

        raise ValueError("--samplingType should belong to %s" % ["samespeaker", "samesequence", "temporalsamespeaker", "sequential", "uniform"])

    def getDataLoader(self, batchSize, type, randomOffset, numWorkers=0,
                      onLoop=-1, nLoops = -1, remove_artefacts=False, batch_size_per_gpu=None):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["speaker", "sequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "speaker": grouped sampler speaker-wise
                type == "sequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                type == "temporalsamespeaker": temporal same speaker sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
            - remove_artefacts : if True, will shift sequences so that no artefact
                                is created (if temporal_sampling is activated)
        """
        totSize = self.totSize // (self.sizeWindow * batchSize)
        if onLoop >= 0:
            self.currentPack = onLoop - 1
            self.loadNextPack()
            nLoops = 1 if nLoops <= 0 else nLoops
        elif nLoops <= 0:
            nLoops = len(self.packageIndex)

        def samplerCall():
            if randomOffset:
                if type == "temporalsamespeaker":
                    # We sample the whole batch at once
                    offset = random.randint(0, self.sizeWindow * batchSize)
                else:
                    # We sample sequence per sequence
                    offset = random.randint(0, self.sizeWindow // 2)
            else:
                offset = 0
            return self.getBaseSampler(type, batchSize, offset, batch_size_per_gpu)

        return AudioLoader(self, samplerCall, nLoops, self.loadNextPack,
                           totSize, numWorkers, remove_artefacts)


def loadFile(data, signal_quality_step=None):
    info1, info2 = data

    if isinstance(info1, int):
        # without signal quality estimations
        seq_info = info1, info2
        signal_quality_path = None
    else:
        seq_info, signal_quality_path = info1, info2

    speaker, fullPath = seq_info
    seqName = fullPath.stem

    # Load audio
    seq = torchaudio.load(str(fullPath))[0].mean(dim=0)
    # Load signal quality estimations
    if signal_quality_path is not None:
        signal_quality = torch.cat(torch.load(signal_quality_path), dim=1)
        seq = seq[:signal_quality.shape[0]*signal_quality_step]
        return speaker, seqName, seq, signal_quality
    return speaker, seqName, seq

class PeakNorm(object):

    def __call__(self, x):
        #Input Size: C x L
        max_val = x.abs().max(dim=1, keepdim=True)[0]
        return x / (max_val + 1e-8)

class AudioLoader(object):
    r"""
    A DataLoader meant to handle an AudioBatchData object.
    In order to handle big datasets AudioBatchData works with big chunks of
    audio it loads sequentially in memory: once all batches have been sampled
    on a chunk, the AudioBatchData loads the next one.
    """
    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers,
                 remove_artefacts=False):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers
        self.remove_artefacts = remove_artefacts

    def __len__(self):
        return self.size

    def get_data_loader(self):
        sampler = self.samplerCall()

        # Remove artefacts
        if self.remove_artefacts:
            sampler = self.__remove_artefacts(sampler)

        return DataLoader(self.dataset,
                          batch_sampler=sampler,
                          num_workers=self.numWorkers)

    def __remove_artefacts(self, sampler):
        """
        Loop through the batches built by the sampler and shift all sequences to remove artefacts.
        Return a sampler object whose batches have been modified.

        If the sampler is an instance of the TemporalSameSpeakerSampler class,
        make sure to shift all the sequences so that we don't create overlap between
        the sequences.
        """
        seqLabels = self.dataset.seqLabel
        windowSize = self.dataset.sizeWindow
        new_batches = []
        for batch in sampler.batches:
            new_batch = []
            # The offset variable will save the number of frames
            # the sequences must be shifted to ensure no overlap
            # is introduced when applying temporalsamespeaker sampling
            offset = 0
            for beg_seq in batch:
                beg_seq += offset
                delete_batch = False
                # Loop through the sequences' name
                for i in range(1, len(seqLabels)):
                    # if there's an artifact
                    if seqLabels[i-1] <= beg_seq < seqLabels[i]:
                        if beg_seq + windowSize > seqLabels[i]:
                            if i != len(seqLabels)-1:
                                new_batch.append(seqLabels[i])
                            else:
                                print("warning, deleting batch because artifact "
                                      "cannot be removed without going out of bounds")
                                delete_batch = True
                            if isinstance(sampler, TemporalSameSpeakerSampler):
                                offset += seqLabels[i] - beg_seq
                        else:
                            new_batch.append(beg_seq)

            if not delete_batch:
                new_batches.append(new_batch)
        sampler.batches = new_batches
        return sampler

    def __iter__(self):
        for i in range(self.nLoop):
            dataloader = self.get_data_loader()
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()

    # Debug functions
    def get_data_loader_verbose(self):
        """
        Verbose version of the get_data_loader function.
        Look for the name of the audio sequences and check if any artefacts
        have been created in the sampling process.

        Should only be used for debug purposes
        """
        def find_audio_name(seqLabels, seqNames, beg_seq):
            artefact_created = False
            for i in range(1, len(seqLabels)):
                if seqLabels[i-1] <= beg_seq < seqLabels[i]:
                    if beg_seq + self.dataset.sizeWindow > seqLabels[i]:
                        artefact_created = True
                    return seqNames[i-1], artefact_created
            raise ValueError("I got beg_seq = %s but my seqLabels is %s " % (beg_seq, seqLabels))

        sampler = self.samplerCall()

        if self.remove_artefacts:
            sampler = self.__remove_artefacts(sampler)

        seqLabels = self.dataset.seqLabel
        seqNames = self.dataset.getSeqNames()
        windowSize = self.dataset.sizeWindow
        sampler_names = []
        sampler_artefacts = []
        for batch in sampler.batches:
            batch_names = []
            artefacts_created = []
            beg_seq_prev = -windowSize
            for beg_seq in batch:
                if beg_seq_prev + windowSize > beg_seq and isinstance(sampler, TemporalSameSpeakerSampler):
                    raise ValueError("Overlap detected [%d,%d] with [%d,%d]" % (beg_seq_prev, beg_seq_prev+windowSize,
                                                                                beg_seq, beg_seq+windowSize))
                batch_name, artefact_created = find_audio_name(seqLabels, seqNames, beg_seq)
                batch_names.append(batch_name)
                artefacts_created.append(artefact_created)
                beg_seq_prev = beg_seq
            sampler_names.append(batch_names)
            sampler_artefacts.append(artefacts_created)

        return DataLoader(self.dataset,
                          batch_sampler=sampler,
                          num_workers=self.numWorkers), sampler_names, sampler_artefacts

    def iter_verbose(self):
        """
        Verbose iter function. Instead of returing the (sequences, labels) tuple, it will return the following :
                    ( (sequences,labels), sequences_names, sequences_has_artefact )
        where :
            sequences_names contain the path to the audio the sequences have been drawn from
            sequences_has_artefact indicates if the sequences contain an artefact (2 sequences from 2
                different recordings that have been concatenated)

        Should only be used for debug purposes
        """
        for i in range(self.nLoop):
            dataloader, sampler_names, sampler_artefacts = self.get_data_loader_verbose()
            for i, x in enumerate(dataloader):
                yield x, sampler_names[i], sampler_artefacts[i]

            if i < self.nLoop - 1:
                self.updateCall()


class UniformAudioSampler(Sampler):

    def __init__(self,
                 dataSize,
                 sizeWindow,
                 offset):

        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset

        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        return iter((self.offset
                     + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len


class SequentialSampler(Sampler):

    def __init__(self, dataSize, sizeWindow, offset, batchSize):

        self.len = (dataSize // sizeWindow) // batchSize
        self.sizeWindow = sizeWindow
        self.offset = offset
        self.startBatches = [x * (dataSize // batchSize)
                             for x in range(batchSize)]
        self.batchSize = batchSize

        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        for idx in range(self.len):
            yield [self.offset + self.sizeWindow * idx
                   + start for start in self.startBatches]

    def __len__(self):
        return self.len


class TemporalSameSpeakerSampler(Sampler):

    def __init__(self,
                 batchSize,
                 samplingIntervals,
                 sizeWindow,
                 offset,
                 batchSizePerGPU=None):
        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow
        self.batchSize = batchSize
        self.offset = offset
        self.batchSizePerGPU = batchSizePerGPU

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1

        # One batch will be of size : self.sizeWindow * self.batchSize
        # And one batch will be made of consecutive chunks of audios
        self.sizeSamplers = [(self.samplingIntervals[i+1] -
                              self.samplingIntervals[i]) // (self.sizeWindow * self.batchSize)
                             for i in range(nWindows)]

        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]

        if sum(self.sizeSamplers) == 0:
            raise ValueError("No sampling intervals can be found. "
                             "Try to increase --max_size_loaded or to reduce the batch size.")
        self.build_batches()

    def __len__(self):
        return len(self.batches)

    def getIndices(self, x, iInterval):
        beg = self.offset + x * self.sizeWindow * self.batchSize \
              + self.samplingIntervals[iInterval]
        # I compared range with np.arange, and the first one is so much faster
        return range(beg, beg + self.sizeWindow * self.batchSize, self.sizeWindow)

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)

    def build_batches(self):
        order = [(x, torch.randperm(val).tolist())
                 for x, val in enumerate(self.sizeSamplers) if val > 0]

        # Build batches (or minibatches depending on the config)
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, len(randperm)
            while indexStart < sizeSampler:
                indexEnd = min(sizeSampler, indexStart + self.batchSize)
                for x in randperm[indexStart:indexEnd]:
                    locBatch = self.getIndices(x, indexSampler)
                    self.batches.append(locBatch)
                indexStart = indexEnd

class SameSpeakerSampler(Sampler):

    def __init__(self,
                 batchSize,
                 samplingIntervals,
                 sizeWindow,
                 offset):

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow
        self.batchSize = batchSize
        self.offset = offset

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [(self.samplingIntervals[i+1] -
                              self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]
        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]
        self.build_batches()

    def __len__(self):
        return len(self.batches)

    def getIndex(self, x, iInterval):
        return self.offset + x * self.sizeWindow \
            + self.samplingIntervals[iInterval]

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)

    def build_batches(self):
        order = [(x, torch.randperm(val).tolist())
                 for x, val in enumerate(self.sizeSamplers) if val > 0]

        # Build Batches
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, len(randperm)
            while indexStart < sizeSampler:
                indexEnd = min(sizeSampler, indexStart + self.batchSize)
                locBatch = [self.getIndex(x, indexSampler)
                            for x in randperm[indexStart:indexEnd]]
                indexStart = indexEnd
                self.batches.append(locBatch)


def extractLength(couple):
    speaker, locPath = couple
    try:
        info = torchaudio.info(str(locPath))
        dur = info.num_frames
    except:
        info = torchaudio.info(str(locPath))[0]
        dur = info.length
    return dur


def findAllSeqs(dirName,
                no_speaker=False,
                extension='.flac',
                loadCache=False,
                speaker_level=1,
                format=None,
                cache_path=None):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers

        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index

        outSpeakers
        The speaker labels (in order)

    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension

    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label should be retrieved no matter the
    organization of the dataset.

    """
    if cache_path is None:
        cache_path = str(Path(dirName) / '_seqs_cache.txt')
    if loadCache:
        try:
            outSequences, speakers = torch.load(cache_path)
            print(f'Loaded from cache {cache_path} successfully')
            return outSequences, speakers
        except OSError as err:
            print(f'Ran in an error while loading {cache_path}: {err}')
        print('Could not load cache, rebuilding')

    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []

    outSequencesIds = []
    outIds = []
    idsTarget= {}

    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]

        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]

            for filename in filtered_files:
                full_path = os.path.join(root[prefixSize:], filename)
                outSequences.append((speaker, full_path))
                if format is not None:
                    # Really need to factor that with a single parameter that points to
                    # a file containing the sorted audios...otherwise it will very
                    # likely cause bugs ...
                    if format == "id_spkr_onset_offset":
                        idStr = '_'.join(filename.split('_')[0:-2])
                    elif format == "id_spkr_onset_offset_spkr_onset_offset":
                        idStr = '_'.join(filename.split('_')[0:-5])
                    elif format == "spkr-id":
                        idStr = '-'.join(filename.split('-')[0:2])
                    elif format == "spkr_id_nb":
                        idStr = '_'.join(filename.split('_')[0:-1])
                    elif format == "spkr-id-nb":
                        idStr = '-'.join(filename.split('-')[0:-1])
                    elif format == "full_seedlings":
                        splitted = filename.split('_')
                        idStr = '_'.join(splitted[0:-2] + [splitted[-1]])
                    else:
                        if format != "no_speaker":
                            raise ValueError("%f format unknown" % format)

                    if format == "no_speaker" or no_speaker:
                        idStr = 'anonymous'
                    if idStr not in idsTarget:
                        idsTarget[idStr] = len(idsTarget)
                        outIds.append(idStr)
                    outSequencesIds.append((idsTarget[idStr], full_path))

    outSpeakers = [None for x in speakersTarget]
    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    # For temporal sampling, files need to be sorted temporally
    if format is not None:
        # We sort by onset
        def get_id_spkr_onset(x):
            # Returns (id_spkr, onset) tuple
            filename = x[1]
            splitted = filename.split('_')
            return '_'.join(splitted[0:-2]), float(splitted[-2])

        def get_id_spkr_onset2(x):
            # Returns (id_spkr, onset) tuple
            filename = x[1]
            splitted = filename.split('_')
            return '_'.join(splitted[0:-5]), float(splitted[-5])

        def get_spkr_id(x):
            # Returns (spkr, id) tuple
            filename = x[1]
            splitted = filename.split('-')
            return splitted[0], int(splitted[1])

        def get_spkr_id2(x):
            # Returns (spkr, id) tuple
            filename = x[1].replace(extension, '')
            splitted = filename.split('_')
            return splitted[0:-1], int(splitted[-1])

        def get_spkr_id3(x):
            # Returns (spkr, id) tuple
            filename = x[1].replace(extension, '')
            splitted = filename.split('-')
            return splitted[0:-1], int(splitted[-1])

        def get_spkr_id_full_seedlings(x):
            # Returns (spkr, id) tuple
            filename = x[1]
            splitted = filename.split('_')
            return splitted[0:-2] + [splitted[-1]], int(splitted[-2])

        def get_no_speaker(x):
            filename = x[1].replace(extension, '')
            splitted = filename.split('_')
            return splitted[0:-1], int(splitted[-1])

        if format == "id_spkr_onset_offset":
            sorting_func = get_id_spkr_onset
        elif format == "id_spkr_onset_offset_spkr_onset_offset":
            sorting_func = get_id_spkr_onset2
        elif format == "spkr-id":
            sorting_func = get_spkr_id
        elif format == "spkr_id_nb":
            sorting_func = get_spkr_id2
        elif format == "spkr-id-nb":
            sorting_func = get_spkr_id3
        elif format == "full_seedlings":
            sorting_func = get_spkr_id_full_seedlings
        elif format == "no_speaker":
            sorting_func = get_no_speaker
        else:
            raise ValueError("can't find sorting func from %s" % format)
        outSequencesIds = sorted(outSequencesIds, key=sorting_func)
        if format == "no_speaker" or no_speaker:
            outSequencesIds = [(0,v) for _, v in outSequencesIds]
        outSequences = outSequencesIds
        outSpeakers = outIds
    try:
        torch.save((outSequences, outSpeakers), cache_path)
        print(f'Saved cache file at {cache_path}')
    except OSError as err:
        print(f'Ran in an error while saving {cache_path}: {err}')
    return outSequences, outSpeakers


def parseSeqLabels(pathLabels):
    with open(pathLabels, 'r') as f:
        lines = f.readlines()
    output = {"step": 160}  # Step in librispeech dataset is 160bits
    maxPhone = 0
    for line in lines:
        data = line.split()
        output[data[0]] = [int(x) for x in data[1:]]
        maxPhone = max(maxPhone, max(output[data[0]]))
    return output, maxPhone + 1


def filterSeqs(pathTxt, seqCouples):
    with open(pathTxt, 'r') as f:
        inSeqs = [p.replace('\n', '') for p in f.readlines()]

    inSeqs.sort()
    seqCouples.sort(key=lambda x: os.path.basename(os.path.splitext(x[1])[0]))
    output, index = [], 0
    for x in seqCouples:
        seq = os.path.basename(os.path.splitext(x[1])[0])
        while index < len(inSeqs) and seq > inSeqs[index]:
            index += 1
        if index == len(inSeqs):
            break
        if seq == inSeqs[index]:
            output.append(x)
    return output
