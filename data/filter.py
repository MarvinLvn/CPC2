"""
Creates a symlinks of the seedlings segments filtered
at 10, 20, 30, 40, 50, 60, 70, 80, and 90% for snr, c50, snr_c50

Use-case

If the dataframe with the snr and c50 scores is already created :
python filter.py path/to/segments/wavfiles -p [percentage_to_filter] -c criterion --table path/to/snr_c50_table

If the dataframe is not created :
python filter.py path/to/segments/wavfiles -p [percentage_to_filter] -c criterion --create_pred_table path/to/predictions

"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn import preprocessing


def create_snr_c50_table(segment_dir, pred_dir):
    """
    Creates and saves a dataframe with all segments,
    and the correspoding C50 score, SNR score and the
    mean of normalized C50 and SNR
    """
    segment_dir = Path(segment_dir) / 'no_filter'
    pred_dir = Path(pred_dir)

    snr_df = pd.read_csv(pred_dir / 'mean_snr_labels.txt', sep=" ", header=None, names=['uri', 'snr'])
    c50_df = pd.read_csv(pred_dir / 'reverb_labels.txt', sep=" ", header=None, names=['uri', 'c50'])
    scores_df = pd.merge(snr_df, c50_df, on='uri')
    wav_df = pd.DataFrame({'path': list(segment_dir.glob('**/*.wav'))})
    wav_df['uri'] = wav_df['path'].map(lambda x: x.stem)
    wav_df['subpath'] = wav_df['path'].map(lambda x: x.relative_to(segment_dir))

    if len(wav_df) != len(scores_df):
        raise ValueError(f"Number of predictions (= {len(scores_df)}) should be equal "
                         f"to number of wav files (= {len(wav_df)}).")

    scores_df = pd.merge(scores_df, wav_df, on='uri')
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized = min_max_scaler.fit_transform(scores_df[["snr", "c50"]])

    scores_df['snr_normalized'] = pd.Series(normalized[:,0])
    scores_df['c50_normalized'] = pd.Series(normalized[:,1])

    scores_df["snr_c50"] = (scores_df["snr_normalized"] + scores_df["c50_normalized"]) / 2
    scores_df.to_csv(os.path.join(segment_dir, 'brouhaha_snr_c50_scores.csv'), sep=',', index=False)
    return scores_df


def filter_data(table, criterion, percentage):
    """
    Returns a dataframe with the top percentage of the criterion
    """
    table_sorted = table.sort_values([criterion], ascending=False)
    number_of_data = int(percentage*table.shape[0]/100)

    files = table_sorted[["uri", "path", "subpath"]][:number_of_data]
    return files


def randomly_filter_data(table, criterion, percentage):
    """
    Returns a dataframe with the top percentage of the criterion
    """
    return table.sample(frac=percentage/100)[["uri", "path", "subpath"]]


def create_symlinks(files, segments_dir, criterion, percentage):
    """
    Creates the symlinks of the top percentage of the segments
    according to the criterion
    """
    segments_dir = Path(segments_dir)
    link_repo = segments_dir / criterion / str(percentage)

    for row in files.iterrows():
        subpath = row[1]['subpath']
        src_path = row[1]['path']
        target_path = src_path
        link_path = link_repo / subpath
        os.makedirs(link_path.parent, exist_ok=True)
        os.symlink(target_path, link_path)


def parse_args(argv):
    """Parser"""
    parser = argparse.ArgumentParser(description='Creates filtered subsets with the top X percents of the dataset'
                                                 'regarding to the desired criterion (snr, c50 or both')

    parser.add_argument('segments_dir', type=str,
                        help="Path to the audio segments")
    parser.add_argument('-p', '--percentage', type=list,
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                        help='List of percentages of desired data filtered. Default: [10, 20, 30, 40, 50, 60, 70, 80, 90]')
    parser.add_argument('-c', '--criterion', type=str,
                        default="all",
                        help='Criterion for the filter. all creates filters for snr, c50 and snr_c50.'
                             'random creates a random subset of the desired percentage'
                             'Default: all',
                        choices=["snr", "c50", "snr_c50", "all", "random"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--create_pred_table', metavar="PREDICTIONS_DIR",
                        help="creates the table with c50 and snr, based on the predictions in PREDICTIONS_DIR")
    group.add_argument('--table', type=str, help="dataframe table with all snr and c50")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show debug information in the standard output")

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.create_pred_table is not None:
        logging.info("Creating the table with snr and c50 scores")
        table = create_snr_c50_table(args.segments_dir, args.create_pred_table)
    else:
        table_csv = args.table
        table = pd.read_csv(table_csv)
    

    if args.criterion=="all":
        logging.info(f"### Creating subsets the following top percentages {args.percentage} regarding to snr, c50, both and random ###")
        for crit in ["snr", "c50", "snr_c50", "random"]:
            for percentage in args.percentage:
                filter = randomly_filter_data if crit == "random" else filter_data
                files = filter(table, crit, percentage)
                create_symlinks(files, args.segments_dir, crit, percentage)
                logging.info(f"Subset of the {percentage} percents top of {crit} done.")

    else:
        logging.info(f"### Creating subsets the top following percentages {args.percentage} regarding to {args.criterion} ###")
        filter = randomly_filter_data if args.criterion == "random" else filter_data
        for percentage in args.percentage:
            files = filter(
                table,
                args.criterion,
                percentage
            )
            create_symlinks(files, args.segments_dir, args.criterion, percentage)
            logging.info(f"Subset of the {percentage} percents top of {args.criterion} done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
