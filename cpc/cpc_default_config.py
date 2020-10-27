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
    group.add_argument('--adversarial', action='store_true',
                       help="(Depreciated) Activate the speaker adversarial "
                       "training.")
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
    group.add_argument('--clustering', type=str, default=None,
                       choices=['deepEmbedded', 'deepClustering',
                                'CTCClustering'],
                       help="(Research) add a clustering loss on top of the "
                       "current training.")
    group.add_argument('--n_clusters', type=int, default=200,
                       help="(Clustering only) Number of clusters to compute")
    group.add_argument('--cluster_delay', type=int, default=0,
                       help="(Clustering only) wait the given number of "
                       "epoch before activating the clustering loss.")
    group.add_argument('--cluster_iter', type=int, default=100,
                       help="(Clustering only) Maximal number of iterations "
                       "when computing the clusters")
    group.add_argument('--clustering_update', type=str, default='kmean',
                       choices=['kmean', 'dpmean'],
                       help="(Clustering only) Clustering method to use.")
    group.add_argument('--multihead_rnn', action='store_true',
                       help="Use one rnn network with k classifiers on top "
                       "of it instead of k independant rnn networks")
    group.add_argument('--adapt_span_loss', type=float, default=2e-6)
    group.add_argument('--transformer_pruning', type=int, default=0)

    group_augment = parser.add_argument_group('Data augmentation configuration',
                                      description="The arguments defining the "
                                      "data augmentation.")
    group_augment.add_argument('--noise_extension', type=str, default='.wav')
    group_augment.add_argument('--augment_future', action='store_true')
    group_augment.add_argument('--augment_past', action='store_true')
    group_augment.add_argument('--augment_type', type=str, choices=['none', 'bandreject', 'pitch',
                                         'pitch_dropout', 'pitch_quick',
                                         'additive', 'reverb', 'time_dropout',
                                         'reverb_dropout'], nargs='+')
    group_augment.add_argument('--bandreject_scaler', type=float, default=1.0)
    group_augment.add_argument('--additive_noise_snr', type=float, default=15.0)
    group_augment.add_argument('--t_ms', type=int, default=100)
    group_augment.add_argument('--pathDBNoise', type=str, default=None)
    group_augment.add_argument('--pathSeqNoise', type=str, default=None)
    group_augment.add_argument('--naming_convention', type=str, default=None, choices=[None, 'id_spkr_onset_offset'])
    group_augment.add_argument('--train_prop', type=float, default=0.9, required=False,
                               help="Proportion of files belonging to the training set"
                                    " (only if pathVal and pathTrain are not specified)")
    group_augment.add_argument('--no_artefacts', action='store_true',
                               help="Avoid creating artefacts when building batches. "
                                    "If this option is activated, it will check for each sequence that the latter "
                                    "remains in one single recording. If not, it will shift the sequence "
                                    "to avoid creating artefacts.")
    group_augment.add_argument('--mask_prob', type=float, default=0.0,
                               help="Probability of creating a mask on the encoded features "
                                    "(only supported for CPC models for now).")
    group_augment.add_argument('--mask_length', type=int, default=10,
                               help="Number of frames a mask will cover "
                                    "(only supported for CPC models for now).")
    group_augment.add_argument('--n_choose_amongst', type=int, default=None,
                               help="Number of sequences that will be first extracted, "
                                    "and whose cosine distance on their speaker embeddings will be computed."
                                    "Then the closest sequences will be chosen to build the batch."
                                    "Should be greater than the batch size.")
    group_augment.add_argument('--concatenate_spkr_emb', action='store_true',
                               help="If True, will concatenate the speaker embeddings to the input"
                                    "of the prediction network")
    return parser
