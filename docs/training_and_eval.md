### How to train a CPC model?

1) To train a domain-specific learner (data augmentation + same pseudo-speaker sampling):
 
```bash
python cpc/train.py --pathDB $PATH_DS_DATA --pathCheckpoint /where/to/store/the/model  --file_extension .wav \
  --n-levels-gru=2 --multihead-rnn --scheduler-ramp=10 --save-step=5 --n-process-loader=1 \
  --max-size-loaded=4000000000 --no-artefacts --nb-epochs=200 --augment-past --augment-type=pitch,artificial_reverb \
  --sampling-type=temporalsamespeaker --naming-convention=id_spkr_onset_offset --no-artefacts
```

where `$PATH_DS_DATA` contains speech segments organized as in the [data preparation](../docs/data_preparation.md) section.

2) To train a domain-general learner (on the whole audio stream):

```bash
python cpc/train.py --pathDB $PATH_DG_DATA --pathCheckpoint /where/to/store/the/model --file_extension .wav \
  --n-levels-gru=2 --multihead-rnn --scheduler-ramp=10 --save-step=5 --n-process-loader=1 \
  --max-size-loaded=4000000000 --no-artefacts --nb-epochs=200 \
  --sampling-type=temporalsamespeaker --naming-convention=no_speaker --no-artefacts
```

where `$PATH_DG_DATA` contains audio segments (speech and non-speech) organized as in the [data preparation](../docs/data_preparation.md) section.

### How to compute the ABX error rate?

You can compute the ABX error rate on the [Zerospeech2017 dataset](https://zerospeech.com/2017/index.html). 
To begin, download the dataset [here](https://download.zerospeech.com/). Then run the ABX evaluation on a given checkpoint with:

```bash
python ABX.py from_checkpoint $PATH_CHECKPOINT $PATH_ITEM_FILE $DATASET_PATH --seq_norm --strict --file_extension .wav --out $PATH_OUT
```
Where:
- $PATH_CHECKPOINT is the path pointing to the checkpoint to evaluate
- $PATH_ITEM_FILE is the path to the .item file containing the triplet annotations
- $DATASET_PATH path to the directory containing the audio files
- $PATH_OUT path to the directory into which the results should be dumped
- --seq_norm normalize each batch of features across the time channel before computing ABX
- --strict forces each batch of features to contain exactly the same number of frames.

### Bonus: Multi-machines training using slurm

You can train CPC on 2 machines of 4 GPUs each with the following bash script (let's call this script `train_CPC_multi_machines.sh`:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4       
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00

export MASTER=`hostname`
export MASTER_PORT=13369

srun python CPC2/cpc/train.py --distributed --master_port $MASTER_PORT --pathDB /path/to/training/set --pathCheckpoint /where/to/store/the/model ...
```

Then the following line will submit the job:

```bash
sbatch -o my_first_model.txt train_CPC_multi_machines.txt
```

### Bonus: How to train a K-means model from CPC representations?

```bash
python cpc/criterion/clustering/clustering_script.py \
    --pathDB path/to/librispeech/train-clean-100/ \
    --nClusters 50 --MAX_ITER 150 --level_gru 2 \
    --save --load --batchSizeGPU 500 \
    checkpoints/CPC_big_6kh/checkpoint_32.pt \
    checkpoints/clustering_CPC_big_kmeans50/clustering_CPC_big_kmeans50.pt
```

The two last parameters are the path to the pre-trained CPC checkpoint, and the output path of the K-means .pt file that will be generated.

NOTE: This command was done on a P100 16GB GPU, and the batchSizeGPU should be modified according to nClusters, so as to fit the memory. Here are the recommended numbers:

  nClusters  | 20  | 50  | 100 | 200 | 500 | 2000
-------------|-----|-----|-----|-----|-----|------
batchSizeGPU | 500 | 500 | 300 | 200 | 100 |  50