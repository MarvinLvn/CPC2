# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################
# You will find here an example of how to run the distributed mode
# on the FAIR cluster. This is extremly useful for big datasets
#####################################################################

import submitit
import os
from pathlib import Path

#####################################################################
JOB_NAME = "CPC_360_v2"
SLURM_LOGS_DIR = Path.home() / "checkpoints" / JOB_NAME
#CHECKPOINT_DIR = Path().resolve().parent / JOB_NAME
CHECKPOINT_DIR = "/private/home/mriviere/FairInternal/CPC_torch/Librispeech360/channel_norm_attention_pred_dropout_2levels_multihead"
PATH_DB = "/datasets01_101/librispeech/062419/train-clean-360/"

#####################################################################

os.makedirs(SLURM_LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

args = ['--dropout',
        '--hiddenEncoder', '256',
        '--pathCheckpoint', str(CHECKPOINT_DIR),
        '--rnnMode', "transformer",
        '--samplingType', "samespeaker",
        '--nLevelsGRU', '2',
        '--save_step', '5',
        '--multihead_rnn',
        '--schedulerRamp', '5',
        '--distributed',
        '--nGPU', '1',
        '--restart',
        '--ignore_cache',
        '--random_seed', '1856183287',
        '--master_port', '18362',
        '--file_extension', ".flac",
        '--max_size_loaded', '2000000000',
        '--batchSizeGPU', '16',
        '--pathDB', PATH_DB]

# submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder=str(SLURM_LOGS_DIR))
executor.update_parameters(timeout_min=60 * 24 * 3, mem_gb=128,
                           gpus_per_node=8, tasks_per_node=2, nodes=4,
                           partition="learnfair",
                           comment='Neurips: CPC', name=JOB_NAME)



def main(args):
    import sys
    sys.path.append('..')
    import train
    return train.main(args)


job = executor.submit(main, args)
print(f"Slurm job submitted. ID: {job.job_id}")
