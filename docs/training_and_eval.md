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

### Signal-quality aware loss

(Currently works with Alodie's repo; needs to integrate her code)
First, let us predict signal quality measures of the training set:

```bash
python CPC_torch/cpc/eval/inference_vad.py --pathDB $PATH_DB \
    --pathPretrained model/CPC_libribig_600_noise_smooth_reverb_reverse_FT_librispeech/checkpoint_24.pt --pathOut $PATH_PREDICTIONS --file_extension .wav \
    --ignore_cache --hiddenGar 512 --hiddenEncoder 512 --window_shift 160 --no_sentence_level --no_speech 
```

where `$PATH_DB` is the path to the training set, and `$PATH_PREDICTIONS` is the folder where to store predictions. The command above will create `.pt` files, with each line of a `.pt` file being the predicted snr (first column) and the 
predicted c50 (second column) for a 100ms frame.

Once this has been done, we can train the model using the signal-quality aware loss by running:

```bash
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
