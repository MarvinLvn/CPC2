## Data preparation

### Format description


The training set should be arranged as follows:

```
PATH_AUDIO_FILES  
│
└───spkr1
│   └───...
│         │   seq_11.wav
│         │   seq_12.wav
│         │   ...
│   
└───spkr2
    └───...
          │   seq_21.wav
          │   seq_22.wav
```

The `spkrN` information can then be used to sample sequences when training the model (within- or across- sampling)

### Pre-filtering strategies

#### 1. Remove non-speech segments

If you want to discard non-speech segments from your training set, you can use:
- The [voice type classifier](https://github.com/MarvinLvn/voice-type-classifier) if your dataset consists of long-form recordings 
- However, a pretrained [pyannote.audio voice activity detection model](https://github.com/pyannote/pyannote-audio) will likely return more accurate segments if your data is clean

Both return `.rttm` files indicating where there is speech in each audio file. If you used the voice type classifier (as I did) you can run:

```bash
python data/extract_segments.py --audio_path /path/to/audio --rttm_path /path/to/rttm --output_path /path/to/output \
  --durations [128] --sampling random --classes [MAL,FEM] --min_dur 1.5
```

This will create a training set of 128h of audio with segments produced by male and female speakers whose duration is at least 1.5s.
Then, you can run:

```bash
python data/build_power_two_training.py --audio_path /path/to/speech/segments --output_path=/path/to/output \
  --nb_packets=16 --duration=28800
```

This will create 16 training sets of 8 hours (28800 seconds), merged two by two to create 8 training sets of 16 hours, ... until a single training set of 128 hours is obtained.


#### 2. Remove noisy and/or reverberated segments 

If you want to discard noisy and/or reverberated segments from your training set, you can use [Brouhaha](https://github.com/marianne-m/brouhaha-vad).

