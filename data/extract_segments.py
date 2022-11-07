import argparse
import os, sys
import glob
import time
import numpy as np
import subprocess
from operator import itemgetter


def load_all_rttm(rttm_path, classes, regex, min_dur, path_audios):
    """
    This function loads all segments of the rttm files lying in 'rttm_path'
    whose speaker belongs to the 'classes' argument

    For a comparative study of functions loading .csv files, see :
    https://medium.com/casual-inference/the-most-time-efficient-ways-to-import-csv-data-in-python-cc159b44063d
    """
    print("Loading rttm files.")
    t0 = time.time()
    all_segments = []
    nb_files = 0
    for rttm_file in glob.glob(os.path.join(rttm_path, '*'+regex+'*.rttm')):
        # Retrieve the associated audio file
        audio_path = os.path.join(path_audios, os.path.basename(rttm_file).replace(".rttm", ".wav"))
        if os.path.isfile(audio_path):
            nb_files += 1
            with open(rttm_file, 'r') as csv_file:
                for line in csv_file:
                    splitted = line.split(' ')
                    onset = float(splitted[3])
                    duration = float(splitted[4])
                    spkr = splitted[7]
                    if spkr in classes and duration >= min_dur:
                        all_segments.append([audio_path, onset, duration, spkr])
    t1 = time.time() - t0
    print("Found %d .rttm files" % nb_files)
    print("Loaded %d segments in %.2f sec" % (len(all_segments), t1))
    return all_segments


def cut_wave_file(audio_file, onset, duration, spkr, output_path):
    """
    This function cut 'audio_file' from 'onset' to 'onset' + 'duration'.
    Stores the chunk in the .wav format in the 'output_path' directory.
    Naming convention is basename_spkr_onset_offset.wav.
    """
    basename = os.path.basename(audio_file).replace(".wav", "")
    basename = basename + "_%s_%.2f_%.2f.wav" % (spkr, float(onset), float(onset)+float(duration))
    output_path = os.path.join(output_path, spkr, basename)

    cmd = ['sox', audio_file, output_path,
           'trim', str(onset), str(duration)]
    subprocess.call(cmd)


def segment_sampler(all_segments, durations, type, output_path):
    """
    This function sample segments to create subset dataset of duration 'durations' (list)
    'type' argument must belong to ['random', 'longest']

    If 'type' == 'random', chooses a segment with a probability proportional to its duration
    If 'type' == 'longest', longest segments are chosen first.
    """
    max_dur = max(durations)
    tot_dur_seg = sum([seg[2] for seg in all_segments])
    if tot_dur_seg < max_dur:
        raise ValueError("You've asked to extract segments whose cumulated duration would be %d hours.\n"
                         "But all the segments found have a cumulated duration of %s hours." % (max_dur//3600, tot_dur_seg//3600))

    spkrs = np.unique([seg[3] for seg in all_segments])
    # Create necessary directories
    for duration in durations:
        for spkr in spkrs:
            dir_path = os.path.join(output_path, str(duration // 3600) + "h", spkr)
            os.makedirs(dir_path)

    if type == 'random':
        uniform_segment_sampler(all_segments, durations, output_path)
    elif type == 'longest':
        longest_segment_sampler(all_segments, durations, output_path)
    else:
        raise ValueError("Only 'uniform' or 'longest' type of sampler is accepted.")


def uniform_segment_sampler(all_segments, durations, output_path):
    """
    Chooses a segment with a probability proportional to its duration
    """
    cum_dur = 0

    # For now, the output directory is the one with the minimal duration
    output_dir = os.path.join(output_path, str(min(durations)//3600)+"h")

    probabilities = np.asarray([seg[2] for seg in all_segments])
    probabilities = probabilities / sum(probabilities)
    all_segments = np.asarray(all_segments)

    while cum_dur < min(durations) and len(all_segments) != 0:
        # We choose a segment with prob. proportional to its duration
        index_choice = np.random.choice(len(all_segments), size=1, p=probabilities)[0]

        # Create the chunk
        chosen_segment = all_segments[index_choice]
        cut_wave_file(audio_file=chosen_segment[0], onset=float(chosen_segment[1]),
                      duration=float(chosen_segment[2]),
                      spkr=chosen_segment[3],
                      output_path=output_dir)

        cum_dur += float(chosen_segment[2])

        # We remove the choice from our list and update probabilities
        probabilities = np.delete(probabilities, index_choice)
        all_segments = np.delete(all_segments, index_choice, axis=0)

        probabilities = probabilities / sum(probabilities)

        if cum_dur >= min(durations) and len(durations) != 1:
            # Update duration and output_path
            print("Done creating %s h version" % min(durations))
            durations = np.delete(durations, np.where(durations == min(durations)))
            output_dir = os.path.join(output_path, str(min(durations) // 3600) + "h")


def longest_segment_sampler(all_segments, durations, output_path):
    """
    Longest segments are chosen first.
    /!\ DETERMINISTIC SAMPLER
    """
    # Sort segments by durations
    all_segments = sorted(all_segments, key=lambda x: -x[2])

    output_dir = os.path.join(output_path, str(min(durations) // 3600) + "h")
    cum_dur = 0
    for chosen_segment in all_segments:
        # Cut current segment
        cut_wave_file(audio_file=chosen_segment[0], onset=float(chosen_segment[1]),
                      duration=float(chosen_segment[2]),
                      spkr=chosen_segment[3],
                      output_path=output_dir)

        # Update cum dur
        cum_dur += float(chosen_segment[2])

        # Update output_dir when the provided duration is reached
        if cum_dur >= min(durations) and len(durations) != 1:
            # Update duration and output_path
            print("Done creating %s h version" % min(durations))
            durations = np.delete(durations, np.where(durations == min(durations)))
            output_dir = os.path.join(output_path, str(min(durations) // 3600) + "h")


def create_symlink(output_path, durations, classes):
    """
    Once the segments have been created, we create symlink
    to complete the 'biggest' subset.

    If 'durations' == [100, 200, 500] :
    i) This will create symlink of all .wav files belonging to the folder '100h'
    to '200h' and '500h'
    ii) It will create symlink of all .wav files belonging to the folder '200h'
    to '500h'.
    """
    # We can't loop in order because we'd first symlink files from first to second folder
    # And then resymlink second folder to third folder (with second folder having already the symlinks)
    for duration in np.flip(durations):
        greater_durations = [dur for dur in durations if dur > duration]
        for greater_dur in greater_durations:
            for spkr in classes:
                input_folder = os.path.join(output_path, str(duration // 3600) + "h", spkr)
                output_folder = os.path.join(output_path, str(greater_dur // 3600) + "h", spkr)
                for input_file in glob.glob(os.path.join(input_folder, "*.wav")):
                    output_file = os.path.join(output_folder, os.path.basename(input_file))
                    os.symlink(os.path.abspath(input_file),
                               os.path.abspath(output_file))


def main(argv):
    parser = argparse.ArgumentParser(description='This scripts extracts audio segments (.wav)'
                                                 'according to their annotations (.rttm)')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to the directory containing the audio files (.wav format)')
    parser.add_argument("--rttm_path", type=str, required=True,
                        help='Path to the directory containing annotations (.rttm).')
    parser.add_argument("--classes", nargs='+', type=str, required=True,
                        help='List of labels whose segments must be extracted. '\
                             'Those should be amongst KCHI, CHI, MAL, FEM, SPEECH if your annotations'\
                             'have been returned by the voice type classifier.')
    parser.add_argument("--durations", nargs='+', type=int, required=True,
                        help='List of cumulated segment durations (in hours) that you would like to extract.'
                             'Please note that if [100,300] are asked. The extracted dataset of 300 hours will'
                             'contain all the segments belonging to the 100h long version. ')
    parser.add_argument("--sampling", type=str, required=True, choices=['random', 'longest'],
                        help="Define the way the program samples segments. If random is selected, the script "
                             "chooses a segment with a probability proportional to its duration. "
                             "If longest is selected, longest segments are chosen first.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="The folder where the extracted dataset will be stored")
    parser.add_argument('--regex', type=str, required=False, default='Bergelson',
                        help="Regex that must be matched by rttm filenames.")
    parser.add_argument('--min_dur', type=float, required=False, default=0,
                        help="The minimal duration of segments. "
                             "All segments whose duration are lower than --min_dur"
                             "will be ignored.")
    args = parser.parse_args(argv)

    print("Extracting %s hours of %s segments from %s" % (args.durations, args.classes, os.path.basename(args.audio_path)))

    if os.path.isdir(args.output_path):
        raise ValueError("%s already exists" % args.output_path)
    else:
        os.makedirs(args.output_path)

    all_segments = load_all_rttm(rttm_path=args.rttm_path, classes=args.classes,
                                 regex=args.regex, min_dur=args.min_dur,
                                 path_audios=args.audio_path)
    FEM_dur = np.sum([seg[2] for seg in all_segments if seg[3] == 'FEM'])
    MAL_dur = np.sum([seg[2] for seg in all_segments if seg[3] == 'MAL'])
    print("FEM_dur : %.2f" % (FEM_dur/3600))
    print("MAL_dur : %.2f" % (MAL_dur/3600))
    print("TOT_dur : %.2f" % ((FEM_dur+MAL_dur)/3600))
    durations = np.asarray([dur * 3600 for dur in args.durations]) # convert in seconds
    segment_sampler(all_segments=all_segments, durations=durations, type=args.sampling, output_path=args.output_path)
    create_symlink(output_path=args.output_path, durations=durations, classes=args.classes)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)
