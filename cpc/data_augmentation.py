from typing import Tuple
import random
import torch
import torchaudio
import numpy as np
from copy import deepcopy
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
import augment.sox_effects as sox_effects
from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse
from cpc.dataset import findAllSeqs
import os

energy_normalization = lambda wav: wav / (torch.sqrt(torch.mean(wav ** 2)) + 1e-8)
peak_normalization = lambda wav: wav / (wav.abs().max(dim=1, keepdim=True)[0] + 1e-8)

class BandrejectAugment:
    def __init__(self, scaler=1.0):
        self.target_info = {'channels': 1,
                            'length': 0,
                            'precision': 16,
                            'rate': 16000.0,
                            'bits_per_sample': 16}

        def random_band():
            low, high = BandrejectAugment.generate_freq_mask(scaler)
            return f'{high}-{low}'

        self.effect = sox_effects.EffectChain().sinc(
            '-a', '120', random_band).dither()

    @staticmethod
    def freq2mel(f):
        return 2595. * np.log10(1 + f / 700)

    @staticmethod
    def mel2freq(m):
        return ((10.**(m / 2595.) - 1) * 700)

    @staticmethod
    def generate_freq_mask(scaler):
        sample_rate = 16000.0  # TODO: configurable
        F = 27.0 * scaler
        melfmax = BandrejectAugment.freq2mel(sample_rate / 2)
        meldf = np.random.uniform(0, melfmax * F / 256.)
        melf0 = np.random.uniform(0, melfmax - meldf)
        low = BandrejectAugment.mel2freq(melf0)
        high = BandrejectAugment.mel2freq(melf0 + meldf)

        return low, high

    def __call__(self, x):
        src_info = {'channels': 1,
                    'length': x.size(1),
                    'precision': 32,
                    'rate': 16000.0,
                    'bits_per_sample': 32}

        y = self.effect.apply(
            x, src_info=src_info, target_info=self.target_info)

        return y


class PitchAugment:
    def __init__(self, quick=False, shift_max=300):
        """
            shift_max {int} -- shift in 1/100 of semi-tone (default: {100})
        """
        random_shift = lambda: np.random.randint(-shift_max, shift_max)
        effect = sox_effects.EffectChain().pitch(random_shift)

        if quick:
            effect = effect.rate("-q", 16000)
        else:
            effect = effect.rate(16000)
        effect = effect.dither()
        self.effect = effect

    def __call__(self, x):
        target_info = {'channels': 1,
                            # it might happen that the output has 1 frame more
                            # by asking for the specific length, we avoid this
                            'length': x.size(1),
                            'precision': 32,
                            'rate': 16000.0,
                            'bits_per_sample': 32}

        src_info = {'channels': 1,
                    'length': x.size(1),
                    'precision': 32,
                    'rate': 16000.0,
                    'bits_per_sample': 32}

        y = self.effect.apply(x, src_info=src_info, target_info=target_info)
        
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        y = y.view_as(x)
        return y


class PitchDropout:
    def __init__(self, T_ms=100, shift_max=300):
        random_shift = lambda: np.random.randint(-shift_max, shift_max)
        effect = sox_effects.EffectChain().pitch(random_shift).rate("-q", 16000).dither()
        effect = effect.time_dropout(max_seconds=T_ms / 1000.0)
        self.effect = effect

    def __call__(self, x):
        target_info = {'channels': 1,
                            # it might happen that the output has 1 frame more
                            # by asking for the specific length, we avoid this
                            'length': x.size(1),
                            'precision': 32,
                            'rate': 16000.0,
                            'bits_per_sample': 32}

        src_info = {'channels': 1,
                    'length': x.size(1),
                    'precision': 32,
                    'rate': 16000.0,
                    'bits_per_sample': 32}

        y = self.effect.apply(x, src_info=src_info, target_info=target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        y = y.view_as(x)

        return y


class ReverbAugment:
    def __init__(self, shift_max=100):
        random_room_size = lambda: np.random.randint(0, shift_max)
        self.effect = sox_effects.EffectChain().reverb(100, 100, random_room_size).channels(1).dither()

    def __call__(self, x):
        src_info = {'channels': 1,
                    'length': x.size(1),
                    'precision': 32,
                    'rate': 16_000,
                    'bits_per_sample': 32}

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 16,
                       'rate': 16_000,
                       'bits_per_sample': 32}
        y = self.effect.apply(x, src_info=src_info,
                              target_info=target_info).view_as(x)
        return y


class AdditiveNoiseAugment:
    def __init__(self, noise_dataset, snr_min, snr_max, batchSize, sampling='uniform'):
        assert noise_dataset and snr_min <= snr_max
        # Parameters
        self.noise_dataset = noise_dataset
        self.sampling = sampling
        self.batchSize = batchSize
        self.snr_min = snr_min
        self.snr_max = snr_max

        # Sequences initialization
        self.update_noise_loader()
        self.get_next_batch()

        # to be deleted
        #self.speech_batch = torch.empty(0)
        #self.noise_batch = torch.empty(0)
        #self.augmented_speech_batch = torch.empty(0)

    def update_noise_loader(self):
        # Artefacts removal not yet available for uniform sampling
        self.noise_data_loader = iter(self.noise_dataset.getDataLoader(self.batchSize, type=self.sampling,
                                                                       randomOffset=True,
                                                                       numWorkers=0, onLoop=-1, nLoops=-1,
                                                                       remove_artefacts=self.sampling != "uniform"))

    def get_next_batch(self):
        try:
            self.current_noise_batch = next(self.noise_data_loader)[0]
        except StopIteration:
            self.update_noise_loader()
            self.current_noise_batch = next(self.noise_data_loader)[0]

    def get_noise_sequence(self):
        nb_remaining_noise_sequences = self.current_noise_batch.shape[0]
        if nb_remaining_noise_sequences == 0:
            self.get_next_batch()
            #suffixe = "with_reverb"
            #torchaudio.save("/private/home/marvinlvn/Downloads/check_audio/noise_batch_%s.wav" % suffixe, self.noise_batch, sample_rate=16000)
            #torchaudio.save("/private/home/marvinlvn/Downloads/check_audio/speech_batch_%s.wav" % suffixe, self.speech_batch, sample_rate=16000)
            #torchaudio.save("/private/home/marvinlvn/Downloads/check_audio/augmented_speech_batch_snr_min_%d_snr_max_%d_%s.wav" % (self.snr_min, self.snr_max,
            #                                                                                                               suffixe), self.augmented_speech_batch,
            #                sample_rate=16000)
            
            #self.speech_batch = torch.empty(0)
            #self.noise_batch = torch.empty(0)
            #self.augmented_speech_batch = torch.empty(0)
            #exit()

        # Get noise sequence
        noise_sequence = self.current_noise_batch[0, 0, ...]
        # Remove it from the batch
        self.current_noise_batch = self.current_noise_batch[1:, ...]

        return noise_sequence

    def __call__(self, x):
        noise = self.get_noise_sequence()

        # Compute weight associated to the noise sequence
        # so that we obtain a target snr
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        a = float(snr) / 20
        noise_rms = 1 / (10 ** a)
        noised = peak_normalization(energy_normalization(x) + energy_normalization(noise) * noise_rms)

        # to be deleted
        #self.speech_batch = torch.cat((self.speech_batch, peak_normalization(x)), axis=1)
        #self.noise_batch = torch.cat((self.noise_batch, peak_normalization(noise)), axis=1)
        #self.augmented_speech_batch = torch.cat((self.augmented_speech_batch, noised), axis=1)

        return noised


class RandomAdditiveNoiseAugment:
    def __init__(self, snr=15):
        self.snr = np.exp(snr * np.log(10) / 10)

    def __call__(self, x):

        alpha = self.snr / x.std()
        noise = torch.randn(x.size(), device=x.device) / alpha
        return x + noise


class ReverbDropout:
    def __init__(self, T_ms=100):
        random_room_size = lambda: np.random.randint(0, 100)
        self.effect = sox_effects.EffectChain() \
            .reverb("50", "50", random_room_size) \
            .channels().dither().time_dropout(max_seconds=T_ms / 1000.0)

    def __call__(self, x):
        src_info = {'channels': 1,
                    'length': x.size(1),
                    'precision': 32,
                    'rate': 16000.0,
                    'bits_per_sample': 32}

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 16,
                       'rate': 16000.0,
                       'bits_per_sample': 32}

        y = self.effect.apply(x, src_info=src_info, target_info=target_info)

        y = y.view_as(x)
        return y


class TimeDropoutAugment:
    def __init__(self, T_ms=100, sr=16000.0):
        self.effect = sox_effects.EffectChain().time_dropout(max_seconds=T_ms / 1000.0)
        self.sr = sr

    def __call__(self, x):
        y = self.effect.apply(x, src_info={'rate': self.sr}, target_info={'rate': self.sr})
        return y


class NaturalReverb:
    def __init__(self, ir_paths, p, batchSize, sr=32000, batch_wise=False):
        self.p = p
        self.sr = sr
        self.batch_wise = batch_wise
        self.count = 0

        if batch_wise:
            # Batch-wise mode
            self.batchSize = batchSize
            self.ir_files, speakers = findAllSeqs(ir_paths, extension=".wav")
            self.ir_files = [os.path.join(ir_paths, data[1]) for data in self.ir_files]

            print("Found %d files for natural reverberation" % len(self.ir_files))

            # Choose impulse response at random
            self.get_new_impulse_response()
        else:
            # Sequence-wise mode
            self.effect = ApplyImpulseResponse(ir_paths=ir_paths,
                                               p=self.p,
                                               sample_rate=self.sr)
            print("Found %d files for natural reverberation" % len(self.effect.ir_paths))

    def get_new_impulse_response(self):
        ir_file = random.choice(self.ir_files)
        self.effect = ApplyImpulseResponse(ir_paths=[ir_file],
                                           p=self.p,
                                           sample_rate=self.sr)

    def __call__(self, x):

        y = peak_normalization(self.effect(x.view(1, 1, -1)).squeeze(0))

        if self.batch_wise:
            self.count += 1
            if self.count == self.batchSize:
                # Counter update, get new impulse response if needed
                self.get_new_impulse_response()
                self.count = 0
        return y


class AugmentCfg:

    def __init__(self, **kwargs):
        self.augment_type = kwargs["type"]
        self.config = {k: i for k, i in kwargs.items() if k != 'type'}

    def __repr__(self):
        return f"{self.augment_type} : \n {self.config}"


class CombinedTransforms:
    """
    Class to handle multiple transformations = different kinds of data augmentation
    """
    def __init__(self, augment_cfgs, **kwargs):
        # Get the list of transformations
        self.transfors_cfgs = [get_augment(x, **kwargs) for x in augment_cfgs]

    def __call__(self, x):
        # Apply transformation one-by-one
        for transform in self.transfors_cfgs:
            if transform is not None:
                x = transform(x)
        return x


def get_augment(augment_type, **kwargs):
    # Not clean, but allows the user to apply different kind of augmentations
    if not augment_type or augment_type == 'none':
        return None
    elif augment_type == 'bandreject':
        return BandrejectAugment(scaler=kwargs['bandreject_scaler'])
    elif augment_type == 'additive':
        if not kwargs['noise_dataset']: raise RuntimeError('Noise dataset is needed for the additive noise')
        return AdditiveNoiseAugment(kwargs['noise_dataset'], kwargs['additive_noise_snr_min'],
                                    kwargs['additive_noise_snr_max'], kwargs['batchSize'],
                                    kwargs['additive_noise_sampling'])
    elif augment_type == 'pitch':
        return PitchAugment(quick=kwargs['pitch_quick'], shift_max=kwargs['shift_max'])
    elif augment_type == 'artificial_reverb':
        return ReverbAugment()
    elif augment_type == 'time_dropout':
        return TimeDropoutAugment(kwargs['t_ms'])
    elif augment_type == 'artificial_reverb_dropout':
        return ReverbDropout(kwargs['t_ms'])
    elif augment_type == 'random_noise':
        return RandomAdditiveNoiseAugment(kwargs['additive_noise_snr'])
    elif augment_type == 'pitch_dropout':
        return PitchDropout(kwargs['t_ms'], shift_max=kwargs['shift_max'])
    elif augment_type == 'natural_reverb':

        return NaturalReverb(ir_paths=kwargs['pathImpulseResponses'],
                             p=kwargs['impulse_response_prob'],
                             batchSize=kwargs['batchSize'],
                             sr=kwargs['ir_sample_rate'],
                             batch_wise=kwargs['ir_batch_wise'])
    else:
        raise RuntimeError(f'Unknown augment_type = {augment_type}')


def augmentation_factory(args, noise_dataset=None, applied_on_noise=False):
    if applied_on_noise:
        # Meta data augmentation mode
        augment_type = args.meta_aug_type
        ir_batch_wise = args.meta_ir_batch_wise
        if augment_type is not None:
            print("Activating meta data augmentation with : %s" % augment_type)
    else:
        # Standard data augmentation mode
        augment_type = args.augment_type
        ir_batch_wise = args.ir_batch_wise
        print("Activating data augmentation with : %s" % augment_type)

    if not augment_type or augment_type == 'none' or not (args.augment_past or args.augment_future):
        return None

    batchSize = args.nGPU * args.batchSizeGPU
    additive_noise_sampling = "temporalsamespeaker" if args.temporal_additive_noise else "uniform"
    if len(augment_type) > 1:
        aug_args = {"bandreject_scaler": args.bandreject_scaler,
                    "pitch_quick": args.augment_type == 'pitch_quick',
                    "t_ms": args.t_ms,
                    "noise_dataset": noise_dataset,
                    "additive_noise_snr_min": args.min_snr_in_db,
                    "additive_noise_snr_max": args.max_snr_in_db,
                    "additive_noise_sampling": additive_noise_sampling,
                    "impulse_response_prob": args.impulse_response_prob,
                    "pathImpulseResponses": args.pathImpulseResponses,
                    "ir_sample_rate": args.ir_sample_rate,
                    "batchSize": batchSize,
                    "ir_batch_wise": ir_batch_wise,
                    "shift_max": args.shift_max}

        return CombinedTransforms(augment_type, **aug_args)
    else:
        augment_type = augment_type[0]

    if augment_type == 'bandreject':
        return BandrejectAugment(scaler=args.bandreject_scaler)
    elif augment_type in ['pitch', 'pitch_quick']:
        return PitchAugment(quick=args.augment_type == 'pitch_quick', shift_max=args.shift_max)
    elif augment_type == 'artificial_reverb':
        return ReverbAugment()
    elif augment_type == 'time_dropout':
        return TimeDropoutAugment(args.t_ms)
    elif augment_type == 'additive':
        if not noise_dataset: raise RuntimeError('Noise dataset is needed for the additive noise')
        return AdditiveNoiseAugment(noise_dataset, args.min_snr_in_db,
                                    args.max_snr_in_db, batchSize, additive_noise_sampling)
    elif augment_type == 'artificial_reverb_dropout':
        return ReverbDropout(args.t_ms)
    elif augment_type == 'pitch_dropout':
        return PitchDropout(args.t_ms, shift_max=args.shift_max)
    elif augment_type == 'natural_reverb':

        return NaturalReverb(ir_paths=args.pathImpulseResponses,
                             p=args.impulse_response_prob,
                             batchSize=batchSize,
                             sr=args.ir_sample_rate,
                             batch_wise=ir_batch_wise)
    else:
        raise RuntimeError(f'Unknown augment_type = {augment_type}')
