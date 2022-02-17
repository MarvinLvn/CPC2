# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse


def get_default_cpc_config():
    parser = set_default_cpc_config(argparse.ArgumentParser())
    return parser.parse_args([])


def set_default_cpc_config(parser):
    # Run parameters
    group = parser.add_argument_group('Architecture configuration',
                                      description="The arguments defining the "
                                      "model's architecture.")
    group.add_argument('--hiddenEncoder', type=int, default=256,
                       help='Hidden dimension of the encoder network.')
    group.add_argument('--hiddenGar', type=int, default=256,
                       help='Hidden dimension of the auto-regressive network')
    group.add_argument('--nPredicts', type=int, default=12,
                       help='Number of steps to predict.')
    group.add_argument('--negativeSamplingExt', type=int, default=128,
                       help='Number of negative samples to take.')
    group.add_argument('--learningRate', type=float, default=2e-4)
    group.add_argument('--schedulerStep', type=int, default=-1,
                       help='Step of the learning rate scheduler: at each '
                       'step the learning rate is divided by 2. Default: '
                       'no scheduler.')
    group.add_argument('--schedulerRamp', type=int, default=None,
                       help='Enable a warm up phase for the learning rate: '
                       'adds a linear ramp of the given size.')
    group.add_argument('--beta1', type=float, default=0.9,
                       help='Value of beta1 for the Adam optimizer')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='Value of beta2 for the Adam optimizer')
    group.add_argument('--epsilon', type=float, default=1e-08,
                       help='Value of epsilon for the Adam optimizer')
    group.add_argument('--sizeWindow', type=int, default=20480,
                       help='Number of frames to consider at each batch.')
    group.add_argument('--nEpoch', type=int, default=200,
                       help='Number of epoch to run')
    group.add_argument('--samplingType', type=str, default='samespeaker',
                       choices=['samespeaker', 'uniform',
                                'samesequence', 'sequential', 'temporalsamespeaker'],
                       help='How to sample the negative examples in the '
                       'CPC loss.')
    group.add_argument('--nLevelsPhone', type=int, default=1,
                       help='(Supervised mode only). Number of layers in '
                       'the phone classification network.')
    group.add_argument('--cpc_mode', type=str, default=None,
                       choices=['reverse', 'bert', 'none'],
                       help='Some variations on CPC.')
    group.add_argument('--encoder_type', type=str,
                       choices=['cpc', 'mfcc', 'lfb'],
                       default='cpc',
                       help='Replace the encoder network by mfcc features '
                       'or learned filter banks')
    group.add_argument('--normMode', type=str, default='layerNorm',
                       choices=['instanceNorm', 'ID', 'layerNorm',
                                'batchNorm'],
                       help="Type of normalization to use in the encoder "
                       "network (default is layerNorm).")
    group.add_argument('--onEncoder', action='store_true',
                       help="(Supervised mode only) Perform the "
                       "classification on the encoder's output.")
    group.add_argument('--random_seed', type=int, default=None,
                       help="Set a specific random seed.")
    group.add_argument('--speakerEmbedding', type=str, default=None,
                       help="(Optional) Path to the frame-level speaker embeddings files that will "
                            "be fed to the prediction network with "
                            "speaker embeddings along with the usual sequence (.npy format) or"
                            "that will be used to build the batches")
    group.add_argument('--speakerEmbeddingStep', type=int, default=160,
                       help="Step used for the speaker embeddings in number of frames. "
                            "Should be equal to the step used in the encoded representations. "
                            "Default to 160 frames = 10 ms (if --speakerEmbedding is activated)")
    group.add_argument('--size_speaker_emb', type=int, default=512,
                       help="Feature size of the speaker embedding (if --speakerEmbedding is activated)")
    group.add_argument('--dout_speaker_emb', type=int, default=0,
                       help="If > 0, will add a linear layer on top of the speaker embeddings of size "
                            "--dout_speaker_emb before concatenating the resulting representations "
                            "to the context-dependent representations (c). If == 0, will concatenate "
                            "the speaker embeddings directly to the context-dependent reprensentations."
                            "(if --speakerEmbedding is activated).")
    group.add_argument('--arMode', default='LSTM',
                       choices=['GRU', 'LSTM', ' ', 'no_ar', 'transformer'],
                       help="Architecture to use for the auto-regressive "
                       "network (default is lstm).")
    group.add_argument('--nLevelsGRU', type=int, default=1,
                       help='Number of layers in the autoregressive network.')
    group.add_argument('--rnnMode', type=str, default='transformer',
                        choices=['transformer', 'RNN', 'LSTM', 'linear',
                                 'ffd', 'conv4', 'conv8', 'conv12',
                                 'transformer_adaptive_span'],
                       help="Architecture to use for the prediction network")
    group.add_argument('--dropout', action='store_true',
                       help="Add a dropout layer at the output of the "
                       "prediction network.")
    group.add_argument('--abspos', action='store_true',
                       help='If the prediction network is a transformer, '
                       'active to use absolute coordinates.')
    group.add_argument('--multihead_rnn', action='store_true',
                       help="Use one rnn network with k classifiers on top "
                       "of it instead of k independant rnn networks")
    group.add_argument('--adapt_span_loss', type=float, default=2e-6)
    group.add_argument('--transformer_pruning', type=int, default=0)
    group.add_argument('--naming_convention', type=str, default=None,
                               choices=[None, 'full_seedlings', 'no_speaker', 'id_spkr_onset_offset', 'spkr-id', 'id_spkr_onset_offset_spkr_onset_offset', 'spkr_id_nb'])
    group.add_argument('--no_artefacts', action='store_true',
                               help="Avoid creating artefacts when building batches. "
                                    "If this option is activated, it will check for each sequence that the latter "
                                    "remains in one single recording. If not, it will shift the sequence "
                                    "to avoid creating artefacts.")
    group.add_argument('--mask_prob', type=float, default=0.0,
                               help="Probability of creating a mask on the encoded features "
                                    "(only supported for CPC models for now).")
    group.add_argument('--mask_length', type=int, default=10,
                               help="Number of frames a mask will cover "
                                    "(only supported for CPC models for now).")
    group.add_argument('--n_skipped', type=int, default=0,
                               help="Number of time steps that will be skipped in the prediction task.")
    group_augment = parser.add_argument_group('Data augmentation configuration',
                                      description="The arguments defining the "
                                      "data augmentation.")
    group_augment.add_argument('--noise_extension', type=str, default='.wav')
    group_augment.add_argument('--augment_future', action='store_true')
    group_augment.add_argument('--augment_past', action='store_true')
    group_augment.add_argument('--augment_type', type=str, choices=['none', 'bandreject', 'pitch',
                                         'pitch_deropout', 'pitch_quick',
                                         'additive', 'artificial_reverb', 'time_dropout',
                                         'artificial_reverb_dropout', 'natural_reverb'], nargs='+')
    group_augment.add_argument('--bandreject_scaler', type=float, default=1.0)
    group_augment.add_argument('--t_ms', type=int, default=100)
    group_augment.add_argument('--pathDBNoise', type=str, default=None)
    group_augment.add_argument('--pathSeqNoise', type=str, default=None)
    group_augment.add_argument('--past_equal_future', action='store_true',
                               help="If activated, will apply the same data augmentation to past and future"
                                    "sequences")
    group_augment.add_argument('--pathImpulseResponses', type=str, default=None)
    group_augment.add_argument('--impulse_response_prob', type=float, default=1.0)
    group_augment.add_argument('--shift_max', type=float, default=300)
    group_augment.add_argument('--min_snr_in_db', type=float, default=5.0)
    group_augment.add_argument('--max_snr_in_db', type=float, default=20.0)
    group_augment.add_argument('--ir_sample_rate', type=int, default=16000,
                               help="Sample rate of the impulse responses. (Default to 32000)")
    group_augment.add_argument('--temporal_additive_noise', action='store_true',
                               help="If activated, will sample noise sequences in temporal order.")
    group_augment.add_argument('--meta_aug', action='store_true',
                               help="If activated, will augment noise sequences.")
    group_augment.add_argument('--meta_aug_type',  type=str, choices=['none', 'natural_reverb'], nargs='+',
                               help="Indicates which types of data augmented need to be applied on noise sequences"\
                                    "(from MUSAN or custom databases")
    group_augment.add_argument('--ir_batch_wise', action='store_true',
                               help="If activated, will apply the natural reverb at the batch level"
                                    "(same impulse response for the whole batch)")
    group_augment.add_argument('--meta_ir_batch_wise', action='store_true',
                               help="If activated, will apply the natural reverb on the noise sequences "
                                    "at the batch level (same impulse response for the whole batch)")

    return parser
