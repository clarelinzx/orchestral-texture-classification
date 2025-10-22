# Orchestral Texture Classification
Official implementation for the paper: [*A Comparative Study of Statistical Features and Deep Learning for Orchestral Texture Classification*](https://drive.google.com/file/d/1dU5iiPPTHSYmWv18DIAMukWLtroJHhLm/view?usp=sharing) ([poster](https://drive.google.com/file/d/1xVbw_CZJWL6oSEg1XznqTPYGQOQW8cIZ/view?usp=sharing)), presented in APSIPA ASC 2025. 

## Overview
This repository contains the source code, preprocessing scripts, and model implementations used in our study on symbolic orchestral texture classification.
The task assigns textural role labels (e.g., Melody / Rhythm / Harmony) to each *track-bar unit* (each bar for each instrument/track) in symphonies.

We provide:
* Data preprocessing: piano roll, sequential, and statistical feature extraction
* Multiple modeling approaches: rule-based, deep learning, pre-trained models, and feature-based hybrid systems
* Cross-dataset benchmark between the Orchestration dataset ([Le et al., 2022](https://gitlab.com/algomus.fr/orchestration)) and the S3 dataset ([Lin et al., 2024](https://github.com/iis-mctl/mctl-symphony-dataset)). Here the `metadata.csv` are provided for both datasets; for the whole dataset, please visit and download through links above.

*Note 1: "Piece" throughout this repository refers to a movement of a symphony rather than the entire work.*
*Note 2: Some code are presented as pseudo-code, as several implementations are adapted from previous studies.*

## Repository Structure
```
orchestral-texture-classification/
│
├── data_preprocessing/
│   ├── data_cleaning.py
│   ├── data_preprocessing_for_rf.py
│   └── utils.py
│
├── data_processing/
│   └── PianoRollsDataset.py
│
├── dataset/
│   ├── orchestration_dataset/
│   │   └── metadata.csv
│   └── s3_dataset/
│       └── metadata.csv
│
├── models/
│   ├── deep_models.py
│   ├── midibert_texture.py
│   ├── random_forest.py
│   ├── skyline_method.py
│   ├── trainer.py
│   └── utils.py
│
├── settings/
│   ├── annotations.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── instruments.py
│   ├── orchestration_info.py
│   ├── s3_info.py
│   └── statistical_features.py
│
└── quick_start.ipynb
```

## Data Preprocessing
### 1. Piano Roll Representation
* Based on [Chu & Li (2023)](https://github.com/YaHsuanChu/orchestraTextureClassification)
* Converts MusicXML or MIDI files into piano roll arrays of shape `(tracks, time_steps, pitch_dim)`
* Each sample = 3 consecutive bars × 5 tracks (target + 4 random tracks)
* Used for CNN / LSTM / CRNN / Transformer inputs

### 2. Sequential Representation
* Converts symbolic data into token sequences for Transformer/BERT-based models

### 3. Statistical Feature Extraction
* Adapted from [L. Soum-Fontez et al. (2021)](https://hal.science/hal-03322543)
* Extracts handcrafted features per track-bar and all-bar units.

## Models
### 1. Rule-Based (Skyline Method)
* Heuristic baseline assuming melody = highest pitch line within a bar
* Adapted from Soum-Fontez et al., 2021.

### 2. Deep Learning Models (trained from scratch)

| Model       | Architecture           | Input      | Key details                         |
| :---------- | :--------------------- | :--------- | :---------------------------------- |
| CNN         | 2D conv layers     | Piano roll | best training efficiency            |
| BiLSTM      | 2-layer bidirectional  | Piano roll | highest accuracy, 500 epochs        |
| CRNN        | CNN + BiLSTM hybrid    | Piano roll | captures local & temporal info      |
| Transformer | 2-layer self-attention | Piano roll | captures local & temporal info  |

### 3. Pre-Trained Models

| Model          | Base                | Description                                      |
| :------------- | :------------------ | :----------------------------------------------- |
| **MidiBERT-f** | [Chou et al., 2024](https://github.com/wazenmai/MIDI-BERT)| frozen pre-trained model, linear classifier only |
| **MidiBERT-n** | [Chou et al., 2024](https://github.com/wazenmai/MIDI-BERT) | fully fine-tuned                                 |
| **M2BERT**     | [Wang & Su, 2025](https://github.com/york135/M2BERT)   | fine-tuned with pianoroll prediction objective   |

*(Only partially included here; please refer to the original repositories for full implementation details.)*

### 4. Random Forest (RF) and Hybrid
* RF trained on statistical + instrument features
* CNN+RF: concatenates CNN embeddings with statistical features
* Based on Soum-Fontez et al., 2021 and extended to orchestral datasets


## Evaluation
* Metrics: Accuracy, weighted macro Precision / Recall / F1
* Protocols:
  * Fixed test set (on Orchestration dataset)
  * 18-fold cross-validation (leave-one-piece-out on Orchestration)
  * 4-fold cross-validation (leave-one-symphony-out on S3)
* None-labeled bars excluded to avoid score inflation


## Citation
If you use this code or dataset, please cite:

```
@inproceedings{lin2025texture,
  title={A Comparative Study of Statistical Features and Deep Learning for Orchestral Texture Classification},
  author={Lin, Zih-Syuan and Wang, Jun-You and Su, Li},
  booktitle={the 17th Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  year={2025},
  address={Singapore},
}
```

## Author & Contact
**Zih-Syuan Lin** (McGill University / Academia Sinica)
- E-amil: [zih-syuan.lin@mail.mcgill.ca](mailto:zih-syuan.lin@mail.mcgill.ca), [clarelyn@iis.sinica.edu.tw](mailto:clarelyn@iis.sinica.edu.tw)
- GitHub: [https://github.com/zih-syuan](https://github.com/zih-syuan)
