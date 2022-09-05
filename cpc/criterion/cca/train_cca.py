import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from time import time

import numpy as np
import progressbar
from cpc.dataset import findAllSeqs
from cpc.feature_loader import buildFeature, FeatureModule, loadModel, buildFeature_batch
from sklearn.cross_decomposition import CCA


def readArgs(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args


def writeArgs(pathArgs, args):
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)


def loadFeatureMakerCPC(cp_path, no_batch=False):
    assert cp_path[-3:] == ".pt"
    assert os.path.exists(cp_path), \
        f"CPC path at {cp_path} does not exist!!"

    pathConfig = os.path.join(os.path.dirname(cp_path), "checkpoint_args.json")
    CPC_args = readArgs(pathConfig)

    # Load FeatureMaker
    print("")
    print("Loading CPC FeatureMaker")
    if 'level_gru' in vars(CPC_args) and CPC_args.level_gru is not None:
        intermediate_idx = CPC_args.level_gru
    else:
        intermediate_idx = 0

    model = loadModel([cp_path], intermediate_idx=intermediate_idx)[0]

    # If we don't apply batch implementation, we can set LSTM model to keep hidden units, making better representations
    if no_batch:
        model.gAR.keepHidden = True

    feature_maker = FeatureModule(model, CPC_args.onEncoder)
    feature_maker.eval()
    return feature_maker


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Quantize audio files using CPC Clustering Module. Save the embeddings'
                                                 'under .pt format.')
    parser.add_argument('--path_cp_X', type=str,
                        help='Path to the CPC checkpoint for model X.')
    parser.add_argument('--path_cp_Y', type=str,
                        help='Path to the CPC checkpoint for model Y.')
    parser.add_argument('--path_db', type=str,
                        help='Path to the dataset that we want to learn our CCA on.')
    parser.add_argument('--path_output', type=str,
                        help='Path to the output directory, where we''ll be storing the CCA model')
    parser.add_argument('--n_components', type=int, default=100,
                        help='Output dimension of the CCA model.')
    parser.add_argument('--file_extension', type=str, default=".wav",
                        help="Extension of the audio files in the dataset (default: .wav).")
    parser.add_argument('--max_size_seq', type=int, default=10240,
                        help='Maximal number of frames to consider '
                             'when computing a batch of features (defaut: 10240).')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size used to compute features '
                             'when computing each file (defaut: 8).')
    parser.add_argument('--strict', type=bool, default=True,
                        help='If activated, each batch of feature '
                             'will contain exactly max_size_seq frames (defaut: True).')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                             "debugging purposes.")
    parser.add_argument('--no_batch', action='store_true',
                        help="Don't use batch implementation of when building features.")
    parser.add_argument('--cpu', action='store_true',
                        help="Run on a cpu machine.")
    return parser.parse_args(argv)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    print("=============================================================")
    print(f"Canonical correlation analysis script.")
    print("=============================================================")

    # Find all sequences
    print("")
    print(f"Looking for all {args.file_extension} files in {args.path_db}")
    seqNames, _ = findAllSeqs(args.path_db,
                              speaker_level=0,
                              extension=args.file_extension,
                              loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][1])[1].endswith(args.file_extension):
        print(f"Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(args.path_db,
                                  speaker_level=0,
                                  extension=args.file_extension,
                                  loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")

    # Check if directory exists
    if not os.path.exists(args.path_output):
        print("")
        print(f"Creating the output directory at {args.path_output}")
        Path(args.path_output).mkdir(parents=True, exist_ok=True)
    writeArgs(os.path.join(args.path_output, "CCA_info_args.json"), args)

    # Debug mode
    if args.debug:
        nsamples = 1000
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]

    assert len(seqNames) > 0, \
        "No file to fit the CCA on!"

    feature_maker_X = loadFeatureMakerCPC(args.path_cp_X, args.no_batch)
    feature_maker_Y = loadFeatureMakerCPC(args.path_cp_Y, args.no_batch)
    if not args.cpu:
        feature_maker_X.cuda()
        feature_maker_Y.cuda()

    def cpc_feature_extraction(featureMaker, x):
        if args.no_batch is False:
            return buildFeature_batch(featureMaker, x,
                                      seqNorm=False,
                                      strict=args.strict,
                                      maxSizeSeq=args.max_size_seq,
                                      batch_size=args.batch_size)
        else:
            return buildFeature(featureMaker, x,
                                seqNorm=False,
                                strict=args.strict)
    print("CPC FeatureMaker loaded!")

    # Quantization of files
    print("")
    print(f"Extracting representations ...")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    features_X = []
    features_Y = []
    for index, vals in enumerate(seqNames):
        bar.update(index)
        file_path = vals[1]
        file_path = os.path.join(args.path_db, file_path)

        X_feat = cpc_feature_extraction(feature_maker_X, file_path)
        Y_feat = cpc_feature_extraction(feature_maker_Y, file_path)
        features_X.append(X_feat)
        features_Y.append(Y_feat)
    features_X = np.concatenate(features_X, axis=1)[0]
    features_Y = np.concatenate(features_Y, axis=1)[0]
    bar.finish()
    print(f"...done {len(seqNames)} files in {time() - start_time} seconds.")

    print("Fitting CCA to extracted features ...")
    cca = CCA(n_components=args.n_components)
    cca.fit(features_X, features_Y)
    print("Done learning CCA parameters.")

    cca_path = os.path.join(args.path_output, "cca_model_n_components_%d.pkl" % args.n_components)
    with open(cca_path, 'wb') as file:
        pickle.dump(cca, file)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)