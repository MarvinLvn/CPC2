from typing import Tuple
import random
import torch
import torchaudio
import numpy as np
from copy import deepcopy
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
import augment.effects as sox_effects

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
                                  target_info=target_info)
        y = y.view_as(x)
        return y


class AdditiveNoiseAugment:
    def __init__(self, noise_dataset, snr):
        assert noise_dataset and snr >= 0.0
        self.noise_dataset = noise_dataset
        r = np.exp(snr * np.log(10) / 10)
        self.t = r / (1.0 + r)
        self.update_noise_loader()

    def update_noise_loader(self):
        self.noise_data_loader = iter(self.noise_dataset.getDataLoader(1, 'uniform', True))

    def __call__(self, x):
        #idx = np.random.randint(0, len(self.noise_dataset))
        try:
            noise = next(self.noise_data_loader)[0]
        except StopIteration:
            self.update_noise_loader()
            noise = next(self.noise_data_loader)[0]
        #noise = self.noise_dataset[idx][0]
        # noise is non-augmented, we get two identical samples
        self.t = self.t/2
        noise = noise[0, 0, ...]

        noised = self.t * x + (1.0 - self.t) * noise.view_as(x)

        return noised

class RandomAdditiveNoiseAugment:
    def __init__(self, snr=15):
        self.snr = np.exp(snr * np.log(10) / 10)

    def __call__(self, x):

        alpha = self.snr / x.std()
        noise = torch.randn(x.size(), device = x.device) / alpha
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
    def __init__(self, T_ms=100, sr=16000):
        self.effect = sox_effects.EffectChain().time_dropout(max_seconds=T_ms / 1000.0)

    def __call__(self, x):
        y = self.effect.apply(x, src_info={'rate': 16000.0}, target_info={'rate': 16000.0})
        return y


class AugmentCfg:

    def __init__(self, **kwargs):
        self.augment_type = kwargs["type"]
        self.config = {k:i for k, i in kwargs.items() if k!= 'type'}

    def __repr__(self):
        return f"{self.augment_type} : \n {self.config}"


class CombinedTransforms:

    def __init__(self, augment_cfgs, **kwargs):

        self.transfors_cfgs = [get_augment(x, **kwargs) for x in augment_cfgs]

    def __call__(self, x):
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
        return AdditiveNoiseAugment(kwargs['noise_dataset'], kwargs['additive_noise_snr'])
    elif augment_type == 'pitch':
        return PitchAugment(quick=kwargs['pitch_quick'])
    elif augment_type == 'reverb':
        return ReverbAugment()
    elif augment_type == 'time_dropout':
        return TimeDropoutAugment(kwargs['t_ms'])
    elif augment_type == 'reverb_dropout':
        return ReverbDropout(kwargs['t_ms'])
    elif augment_type == 'random_noise':
        return RandomAdditiveNoiseAugment(kwargs['additive_noise_snr'])
    elif augment_type in ['pitch_dropout']:
        return PitchDropout(kwargs['t_ms'])
    else:
        raise RuntimeError(f'Unknown augment_type = {augment_type}')

def augmentation_factory(args, noise_dataset=None):

    if not args.augment_type or args.augment_type == 'none':
        return None

    if len(args.augment_type) > 1:
        aug_args = {"bandreject_scaler" : args.bandreject_scaler,
                    "pitch_quick": args.augment_type == 'pitch_quick',
                    "t_ms": args.t_ms,
                    "noise_dataset": noise_dataset,
                    "additive_noise_snr": args.additive_noise_snr}
        return CombinedTransforms(args.augment_type, **aug_args)
    else:
        args.augment_type = args.augment_type[0]

    if args.augment_type == 'bandreject':
        return BandrejectAugment(scaler=args.bandreject_scaler)
    elif args.augment_type in ['pitch', 'pitch_quick']:
        return PitchAugment(quick=args.augment_type == 'pitch_quick')
    elif args.augment_type == 'reverb':
        return ReverbAugment()
    elif args.augment_type == 'time_dropout':
        return TimeDropoutAugment(args.t_ms)
    elif args.augment_type == 'additive':
        if not noise_dataset: raise RuntimeError('Noise dataset is needed for the additive noise')
        return AdditiveNoiseAugment(noise_dataset, args.additive_noise_snr)
    elif args.augment_type == 'reverb_dropout':
        return ReverbDropout(args.t_ms)
    elif args.augment_type in ['pitch_dropout']:
        return PitchDropout(args.t_ms)
    else:
        raise RuntimeError(f'Unknown augment_type = {args.augment_type}')
