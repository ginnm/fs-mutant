# FS-mutant: A Few-Shot Learning Dataset for Protein Mutants Mining

<div align="center">
    <a href="https://github.com/ginnmelich/fs-mutant/">
        <img width="200px" height="auto" src="https://github.com/ginnmelich/fs-mutant/blob/main/band.png">
    </a>
</div>

[![GitHub license](https://img.shields.io/github/license/ginnmelich/fs-mutant)](https://github.com/ginnmelich/fs-mutant/blob/main/LICENSE)
Updated on 2023.11.23

## Introduction

Welcome to the official repository of the FS-mutant dataset, a benchmark for few-shot learning in protein mutants mining.

## Download

To download and unzip the dataset, use the following shell commands:
```shell
wget https://lianglab.sjtu.edu.cn/files/FS-Mutant-2023/fs-mutant.zip
unzip fs-mutant.zip
```

## Dataset Structure

The dataset folder contains a curated selection of proteins. To view the structure, use:
```shell
tree dataset -L 1

dataset
├── A4_HUMAN_Seuma_2021
├── CAPSD_AAV2S_Sinai_substitutions_2021
├── DLG4_HUMAN_Faure_2021
├── F7YBW8_MESOW_Aakre_2015
├── GCN4_YEAST_Staller_induction_2018
├── GFP_AEQVI_Sarkisyan_2016
├── GRB2_HUMAN_Faure_2021
├── HIS7_YEAST_Pokusaeva_2019
├── PABP_YEAST_Melamed_2013
├── SPG1_STRSG_Olson_2014
└── YAP1_HUMAN_Araya_2012
```

The main directory, 'dataset', includes subdirectories for various proteins, such as:
- A4_HUMAN_Seuma_2021
- CAPSD_AAV2S_Sinai_substitutions_2021
- DLG4_HUMAN_Faure_2021
- [Other protein directories]

Each protein directory, for example, 'dataset/A4_HUMAN_Seuma_2021', contains:

- A '.fasta file' with the protein sequence.
- A '.scores.csv' file with scores predicted by zero-shot models (sourced from Protein Gym).
- A 'splits' directory for training and testing data.

### Example : A4_HUMAN_Seuma_2021
```shell
tree dataset/A4_HUMAN_Seuma_2021 -L 1

dataset/A4_HUMAN_Seuma_2021
├── A4_HUMAN_Seuma_2021.fasta
├── A4_HUMAN_Seuma_2021.scores.csv
└── splits
```

Within this directory:

- A4_HUMAN_Seuma_2021.fasta
- A4_HUMAN_Seuma_2021.scores.csv
- splits/

The 'splits' directory is organized as follows:

```shell
tree dataset/A4_HUMAN_Seuma_2021/splits -L 1

dataset/A4_HUMAN_Seuma_2021/splits
├── 160_1
├── 160_2
├── 160_3
├── 160_4
├── 160_5
├── 20_1
├── 20_2
├── 20_3
├── 20_4
├── 20_5
├── 320_1
├── 320_2
├── 320_3
├── 320_4
├── 320_5
├── 40_1
├── 40_2
├── 40_3
├── 40_4
├── 40_5
├── 80_1
├── 80_2
├── 80_3
├── 80_4
└── 80_5
```

It contains multiple subdirectories, each named in the format N_I, where N denotes the training set size and I is the split index (e.g., 20_1, 40_2).

### Inside a Split Directory

For instance, in 'dataset/A4_HUMAN_Seuma_2021/splits/20_1/':
```
tree dataset/A4_HUMAN_Seuma_2021/splits/20_1/ -L 1

dataset/A4_HUMAN_Seuma_2021/splits/20_1/
├── compound_multi.csv
├── cross_single.csv
├── local_single.csv
└── train.csv
```

This split directory contains:
- train.csv: The training dataset.
- Other CSV files like compound_multi.csv, cross_single.csv, local_single.csv: These are test datasets.