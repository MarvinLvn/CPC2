import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import random
import soundfile as sf
import subprocess
import sys
import time
from operator import itemgetter
from tqdm import tqdm

random.seed(42)

# python build_power_two_training.py --audio_path /gpfsscratch/rech/xdz/uow84uh/DATA/FRENCH_daylongs_subset_cut --nb_packets=112 --output_path=/gpfsscratch/rech/xdz/uow84uh/DATA/FRENCH_daylongs_subset_no_vad --duration=28800
# + delete 512h/1
# python build_power_two_training.py --audio_path /gpfsscratch/rech/xdz/uow84uh/DATA/ACLEW10K_daylongs_subset_cut --nb_packets=112 --output_path=/gpfsscratch/rech/xdz/uow84uh/DATA/ACLEW10K_daylongs_subset_no_vad --duration=28800
# + delete 512h/1
# same for audiobooks, but no need to deleted as we have enough to create 1024h training sets

# For sliced daylongs :
# python build_power_two_training.py --audio_path $DATA/EN_daylong_sliced_c50_sampling --nb_packets=16 --output_path=$DATA/EN_daylong_vad_high_intelligibility_trn --duration=28800
# python build_power_two_training.py --audio_path $DATA/FR_daylong_sliced_c50_sampling --nb_packets=16 --output_path=$DATA/FR_daylong_vad_high_intelligibility_trn --duration=28800

def get_audio_duration(audio_path):
    f = sf.SoundFile(audio_path)
    return len(f) / f.samplerate


def create_min_dur_packets(audio_path, output_path, target_dur, nb_packets):
    print("Start creating small packets of audio")
    audio_files = glob.glob(os.path.join(audio_path, '**/*.wav'), recursive=True)
    i = 0
    for packet_idx in tqdm(range(0, nb_packets)):
        curr_dur = 0
        packet_path = os.path.join(output_path, str(int(target_dur/3600)) + 'h', '%d' % packet_idx)
        while i < len(audio_files) and curr_dur < target_dur - 0.01 * target_dur:
            audio = audio_files[i]
            base_path = audio.replace(audio_path, '')[1:]
            dest = os.path.join(packet_path, base_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(audio, dest)
            i += 1
            curr_dur += get_audio_duration(audio)
    print("Done creating %d packets of %d hours" % (nb_packets, target_dur//3600))


def gather_small_packets(output_path, target_dur, nb_packets):
    print("Start gathering small packets to create bigger packets")
    while nb_packets > 1:
        for i in range(0, nb_packets, 2):
            path1 = os.path.join(output_path, str(int(target_dur/3600)) + 'h', str(i))
            path2 = os.path.join(output_path, str(int(target_dur/3600)) + 'h', str(i+1))
            files1 = glob.glob(os.path.join(path1, '**/*.wav'), recursive=True)
            files2 = glob.glob(os.path.join(path2, '**/*.wav'), recursive=True)

            packet_path = os.path.join(output_path, str(int(2*target_dur/3600)) + 'h', str(i//2))
            for file in files1 + files2:
                base_path = file.replace(path1, '').replace(path2, '')[1:]
                src = file
                dest = os.path.join(packet_path, base_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.symlink(src, dest)
        nb_packets = nb_packets // 2
        target_dur = target_dur * 2
        print("Done creating %d packets of %d hours" % (nb_packets, target_dur//3600))

def main(argv):
    parser = argparse.ArgumentParser(description='This scripts build smaller mutually exclusive training sets.')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to the directory containing the audio files (.wav format)')
    parser.add_argument("--duration", type=int, required=True, default=8*3600,
                        help='Minimal duration to considered (default to 8 hours)')
    parser.add_argument("--nb_packets", type=int, required=True,
                        help='Number of --duration packets to consider')
    parser.add_argument('--output_path', type=str, required=True,
                        help="The folder where the extracted dataset will be stored")
    args = parser.parse_args(argv)

    if os.path.isdir(args.output_path):
        raise ValueError("%s already exists" % args.output_path)
    else:
        os.makedirs(args.output_path)

    create_min_dur_packets(args.audio_path, args.output_path, args.duration, args.nb_packets)
    gather_small_packets(args.output_path, args.duration, args.nb_packets)

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)