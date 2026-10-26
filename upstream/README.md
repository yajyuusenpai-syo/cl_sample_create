# How to Load Pre-trained Models

```python3
from model import CLAPEncoder

model = CLAPEncoder()
model.load_pretrained()
```

# Training Method

## Step 1. Dataset Preparation

The captions of AudioCaps 2.0 are included as a submodule, so just run the following command:

```bash
git submodule update --init --recursive
cd data
python3 remove_cr.py
```

Place the wav files in `data/wav`.
You can find the download request link on the AudioCaps GitHub page([here](https://github.com/cdjkim/audiocaps/tree/master)).

The RIR dataset is generated via simulation in this project:

```bash
cd data/rir_generator
python3 main.py
```

For event labels used in pre-training, download the labels from the AudioSet([here](https://research.google.com/audioset/index.html)) page and place them under `data/audioset` as follows:

```
data
└── audioset
    ├── balanced_train_segments.csv
    ├── eval_segments.csv
    └── unbalanced_train_segments.csv
```

Then, generate the tag data:

```bash
cd data/event_label
python3 get_info.py
python3 convert_to_tag.py
```

Download monoraul CLAP model:

```bash
mkdir -p data/ckpt
cd data/ckpt
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt
```

## Step 2. Pre-training the Spatial Information Encoder
We pre-train the spatial information encoder using the sound event localization and detection (SELD) task.

```bash
cd pretrain_spatial_encoder
python3 train.py
```

## Step 3. Training CLAP
Next, train CLAP with the following command:

```bash
python3 train.py
```

# Citation
If you use SpatialCLAP in your research, please cite the following paper:

```
@article{seki2025spatial,
  title={Spatial-CLAP: Learning Spatially-Aware audio--text Embeddings for Multi-Source Conditions},
  author={Seki, Kentaro and Okamoto, Yuki and Yamaoka, Kouei and Saito, Yuki and Takamichi, Shinnosuke and Saruwatari, Hiroshi},
  journal={arXiv preprint arXiv:2509.14785},
  year={2025}
}
```
