# Text Region Detection in Historical Astronomical Diagrams 

Official repository of the paper _"Text region detection in historical astronomical diagrams"_. We introduce the first large, diverse, open-access dataset of **948** historical astronomical diagrams annotated with **10,940** oriented polygonal text regions that spans ten centuries (8<sup>th</sup> to 18<sup>th</sup>) and seven major traditions: Arabic, Persian, Chinese, Byzantine, Latin, Hebrew, and Sanskrit.

# Dataset

<p align='center'>
<img src="media/inter-class1.png" height="300pt">
</p>
<p align='center'>
<img src="media/intra-class1.png" height="300pt">
</p>

## Setup 🚧
This project environment is build upon `uv`. To install `uv`, setup and activate the environment run the following commands:
```
# Install to macOS/Linux
curl -LsSf https://astral-sh.uv/install.sh | sh

# Create the environment
uv sync

# Activate the environment
source .venv/bin/activate

# Build MSDA
cd ./models/dino/ops
rm -rf build
uv run python setup.py build install
cd ../../../
```

## Content :scroll:
To download the dataset, replace the data from Hugging Face under `data/` or run `./download_data.sh`.

We provide our dataset under two directories, namely `EIDA` and `EIDALatin`, with annotations in `LabelMe` format. `EIDA` contains images and associated annotations of all traditions (including Latin) under `train`, `val`, and `test` splits:
```
EIDA/
├── train/
│   ├── <filename_1>.jpg
│   ├── <filename_1>.json
│   │   .
│   │   .
│   ├── <filename_N>.jpg
│   └── <filename_N>.json
├── val/
└── test/
```

For class-aware annotations, we provide `EIDALatin`, which contains Latin subset with text classes, and splits in `.txt` format:
```
EIDALatin/
├── data/
│   ├── <filename_1>.jpg
│   ├── <filename_1>.json
│   │   .
│   │   .
│   ├── <filename_N>.jpg
│   └── <filename_N>.json
├── train.txt
├── val.txt
└── test.txt
```
🤗 Hugging Face link to our dataset: [link](https://huggingface.co/datasets/sonatbaltaci/textindiagrams)

# Evaluation :chart_with_upwards_trend:

## Class-agnostic text region detection
To evaluate class-agnostic text region detection results, run:
```
python evaluate.py --mode val \
    --results val_results.json \
    --key_indices 0 K-1 K 2K-1
```
`--key_indices` sets the corner indices for the **reading order** check; `2K` is the number of polygon vertices. The JSON file must be a list of detections, each entry like:
```
{
    "image_name": "<filename>.jpg",
    "segmentation": [x1, y1, x2, y2, ..., xN, yN],
    "score": X
}
```
(`file_name` is accepted instead of `image_name`.) Note that the very first point `[x1, y1]` will be matched with first vertex of the bottom line of a text polygon, left- or right-first, depending on the reading order. For **test** split, pass `--thresholds A B` for F1 and F1-RO at fixed score cutoffs. As an example:
```
python evaluate.py --mode test \
    --results test_results.json \
    --key_indices 0 K-1 K 2K-1 \
    --thresholds F1_optimal_threshold F1-O_optimal_threshold
```

### Per-script analysis
We provide the script splits under `notebook/script_evaluation_map.json`.

## Class-aware text region detection (Latin)
For **class-aware** metrics (per-class matching, mAP, mF1) on the Latin subset, run:
```
python evaluate_class.py --mode val \
    --results val_results.json \
    --config config/latin_20class.py
```
Use `config/latin_19class.py` if your model has 19 classes. The JSON matches the class-agnostic format, with **either** a predicted class id per detection **or** a full score vector:
```
{
    "image_name": "<filename>.jpg",
    "segmentation": [x1, y1, x2, y2, ..., xN, yN],
    "score": X,
    "category_id": <int>
}
```
(`label` is accepted as an alias for `category_id`.) Alternatively, omit `score` / `category_id` and provide `"scores": [p_class0, ..., p_classC-1]` (length `num_classes`). For **test** mode, pass `--thresholds T` so the first value `T` is the score threshold used for reported per-class F1 (default without it follows the Latin notebook). Optional: `--iou_thresh`, `--print_per_class`.

# Poly-DETR Baseline :seedling:
This repository provides the implementation for the proposed baseline, based on [DINO-DETR](https://github.com/IDEA-Research/DINO). The code for generating synthetic diagrams builds on [HDV](https://github.com/vayvi/HDV).

## Training
### Checkpoints
Poly-DETR checkpoints can be found [here](https://drive.google.com/file/d/1jEcmcc0czhJTEoIfsGgiQroIySCXkh0L/view?usp=share_link). To run from a pretrained checkpoint or evaluate our model, move the downloaded checkpoints under `./logs/` directory, or run the following commands:
```
gdown "https://drive.google.com/uc?export=download&id=1jEcmcc0czhJTEoIfsGgiQroIySCXkh0L" -O eida_checkpoints.zip
unzip eida_checkpoints.zip
rm eida_checkpoints.zip
cd ../
```

### Synthetic data resources
To download the synthetic data generation resources, run the following commands:
```
gdown "https://drive.google.com/uc?export=download&id=1CefMh1AHz2vDi9g_kOLsx5GdTcum_WfB" -O synthetic_resource.zip
unzip synthetic_resource.zip
rm synthetic_resource.zip
mv synthetic_resource ./datasets/
```

### Class-agnostic text region detection
To pretrain a class-aware model on Latin subset, run:
```
./scripts/pretraining.sh
```
For finetuning, run:
```
./scripts/finetuning.sh
```

### Class-aware text region detection
To pretrain a class-agnostic model, run:
```
./scripts/latin_pretraining.sh
```
For finetuning, run:
```
./scripts/latin_finetuning.sh
```

## Notebooks
For interactive evaluation, see `notebook/detection_evaluation.ipynb` (class-agnostic) and `notebook/detection_evaluation_latin.ipynb` (class-aware). Command-line equivalents are `evaluate.py` and `evaluate_class.py` above.

# Citation :bookmark:
```
@inproceedings{baltaci2026text,
  title={Text region detection in historical astronomical diagrams},
  author={Baltaci, Zeynep Sonat and Baena, Rapha\"el and Meng, Fei and Norindr, Som and Somer, Florence and Husson, Matthieu and Aubry, Mathieu},
  booktitle={ICDAR},
  year={2026}
}
```

# License
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

# Acknowledgements
This work was funded by the ANR project EIDA ANR-22-CE38-0014, the ANR project VHS ANR-21-CE38-0008, and the ERC project DISCOVER funded by the European Union’s Horizon Europe Research and Innovation program under grant agreement No. 101076028. This work was granted access to the HPC resources of IDRIS under the allocation AD010614956R1 and AD011015222 made by GENCI. The authors would like to thank the many historians and computer vision researchers that contributed to the development of the dataset: Eleonora Andriani (Sphaera project, Max Planck Institute for the History of Sciences, Berlin), Ji Chen, Samuel Guessner, Divna Manolova, Scott Trigg (EIDA project), Malamatenia Vlachou Efstathiou, Léore Bensabath (ENPC), and Vidal Attias (CEA).
