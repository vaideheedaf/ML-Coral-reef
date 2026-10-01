# ML-Coral-reef
# Automated Coral Reef Health Assessment Using Image Processing

A feature-extraction pipeline that turns coral reef photographs into a table of numeric, interpretable features, labelled as **Healthy** or **Bleached**. The output CSVs are ready to be used with classical machine learning algorithms.

This repository covers the data side of the project: fetching the images, extracting features and exporting clean CSV files. Model training is a separate step.

---

## Overview

Coral bleaching happens when corals lose the algae that give them their colour, leaving them pale and chalky. This project captures that change with handcrafted image features instead of a deep learning model, so every number in the dataset has a visual meaning that can be explained.

Each image is described by nine features covering colour, texture, edge structure and information content.

## Dataset

- **Source:** [Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images) on Kaggle (by asfarhossainsitab)
- **Task:** binary classification, `Healthy` vs `Bleached`
- **Structure:**

```
Coral Reef Images/
├── train/
│   ├── Bleached/
│   └── Healthy/
├── test/
│   ├── Bleached/
│   └── Healthy/
└── valid/
    ├── Bleached/
    └── Healthy/
```

The notebook downloads the dataset directly from Kaggle onto the Colab runtime, so nothing needs to be uploaded manually.

## Features

| # | Feature | Colour space | What it captures |
|---|---------|--------------|------------------|
| 1 | `mean_saturation` | HSV | How vivid the colours are. Bleached coral is washed out. |
| 2 | `mean_brightness` | HSV (V channel) | Overall brightness of the image. |
| 3 | `white_pixel_pct` | HSV | Percentage of pixels that are pale and low in saturation. |
| 4 | `mean_lightness` | LAB (L channel) | Perceived lightness. |
| 5 | `lbp_mean` | Grayscale | Micro-texture of the surface (Local Binary Pattern). |
| 6 | `glcm_contrast` | Grayscale | How strongly neighbouring pixels differ (GLCM). |
| 7 | `glcm_homogeneity` | Grayscale | How smooth and uniform the surface is (GLCM). |
| 8 | `edge_density` | Grayscale | Fraction of pixels that are edges (Canny). |
| 9 | `entropy` | Grayscale | How varied the gray levels are (Shannon entropy). |

The colour features come from the colour image. Grayscale is used only for the texture and structure features, which do not need colour.

## Pipeline

1. Download the dataset from Kaggle with `kagglehub`.
2. Locate the folder containing `train`, `test` and `valid`.
3. Read each image and resize it to 256 x 256.
4. Convert to HSV, LAB and grayscale.
5. Compute the nine features.
6. Take the label from the class folder name.
7. Save one CSV per split.

## Output

Three files are produced: `train.csv`, `test.csv` and `valid.csv`.

| Column | Description |
|--------|-------------|
| `image_id` | Sequential identifier, e.g. `train_img1` |
| `image_path` | Path of the original image inside the dataset, e.g. `train/Bleached/<filename>.jpg` |
| `mean_saturation` ... `entropy` | The nine features above |
| `target` | `Healthy` or `Bleached` |

Kaggle does not provide a web link per image, so `image_path` is the pointer back to the original file.

## Getting Started

**Requirements:** Python 3, `kagglehub`, `scikit-image`, `opencv-python-headless`, `pandas`, `numpy`, `matplotlib`, `tqdm`. All of these are installed by the notebook.

**Run on Google Colab**

1. Open `coral_feature_extraction_v2.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. The three CSV files are downloaded at the end.

If the Kaggle download asks for credentials, create an API token from your Kaggle account settings and follow the fallback instructions in the notebook.

## Configuration

These values are set in the feature extraction cell:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `IMG_SIZE` | `(256, 256)` | All images are resized to this so features are comparable. |
| `WHITE_S_MAX` | `60` | A pixel counts as white only if saturation is below this (0 to 255). |
| `WHITE_V_MIN` | `170` | ... and brightness is above this (0 to 255). |

The notebook includes a tuning step for the white-pixel thresholds. It compares several settings on a random sample and shows the resulting mask next to each image. Underwater tint varies, so check the thresholds against your own data.


## Acknowledgements

Dataset by [asfarhossainsitab](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images) on Kaggle. Please check the dataset page for its licence and terms of use.

## Author

Snow, Computer Science & Engineering (Data Science), St. Vincent Pallotti College of Engineering & Technology, Nagpur.
