# How to train a CPC model?

To train a CPC model, use:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
```

# How to compute the ABX error rate?

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

# Bonus: How to train a K-means model from CPC representations?

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