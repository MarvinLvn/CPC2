### How to train a CPC model?

To train a CPC model with data augmentation, and same speaker sampling:
 
```bash
python cpc/train.py --pathDB /path/to/data --pathCheckpoint /path/to/output  --file_extension .wav \
  --n-levels-gru=2 --multihead-rnn --scheduler-ramp=10 --save-step=5 --n-process-loader=1 \
  --max-size-loaded=4000000000 --no-artefacts --nb-epochs=200 --augment-past --augment-type=pitch,artificial_reverb \
  --sampling-type=samespeaker
```

where `/path/to/data` contains audio segments organized as in the [data preparation](../docs/data_preparation.md) section

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
sbatch -o my_first_model.txt train_CPC_multi_machines.sh
```

### Bonus: Signal-quality aware loss (WIP)

You can use [Brouhaha](https://github.com/marianne-m/brouhaha-vad) to predict the Speech-to-Noise Ratio and the C50 of audio segments.


Once this has been done, we can train the model using the signal-quality aware loss by running:

```bash
# Needs to be updated with Brouhaha code
python CPC_audio/cpc/train.py --pathDB $PATH_DB --pathCheckpoint $PATH_OUT --file_extension .wav --n_process_loader 1 --save_step 5 \
  --schedulerRamp 10 --nLevelsGRU 2 --augment_past --augment_type pitch artificial_reverb --shift_max 300 \
  --multihead_rnn --samplingType samespeaker --nEpoch 200 \
  --signal_quality_path $PATH_PREDICTIONS --signal_quality_mode $SIGNAL_QUALITY_MODE --growth_rate $GROWTH_RATE --inflection_point_x $INFLECTION_POINT_X
```

where:
    - `$PATH_OUT` is the folder where checkpoints will be stored. 
    - `$SIGNAL_QUALITY_MODE` belongs to [snr,c50,snr_c50] and indicates which signal quality measure to use. Measures are min-max scaled between 0 and 1.
    - `$GROWTH_RATE` is the measure of the growth rate to use in the sigmoid weighting function (100 --> sharp profile, i.e the model won't learn on noisy segments, 10 --> smooth profile, i.e. the model will learn a bit on noisy segments)
    - `$INFLECTION_POINT_X` corresponds to the center of the sigmoid weighting function (knowing that the signal quality measures lies in [0, 1]). Default to 0.5.

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
