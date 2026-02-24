# 👁️ Glaucoma Screening via Retinal Image Segmentation

An automated image processing pipeline designed to segment the **Optic Disc** and **Optic Cup** from fundus photographs. Optimized using **Connected Component Analysis (CCA)** and adaptive **V-set selection** for high-precision glaucoma screening.

> [!IMPORTANT]
> The **Drishti-GS** dataset used in this project is a public benchmark. You can download the raw data from the official source or the link provided in the project documentation. This repository contains the source code, experimental methodology, and segmentation results.

**Key Achievements**:

* ✅ **Automated V-set Design**: High-accuracy intensity thresholding derived from training data.
* ✅ **8-Connectivity Optimization**: Robust component labeling to eliminate vascular noise.
* ✅ **Adaptive FOV Masking**: Eliminates non-retinal artifacts and black camera borders.
* ✅ **Dice Coefficient Validation**: Rigorous overlap analysis against expert manual annotations.

---

## 📖 Overview

The **Cup-to-Disc Ratio (CDR)** is a vital clinical metric for diagnosing Glaucoma. This project develops a non-deep-learning pipeline that utilizes classical Digital Image Processing techniques to isolate the Optic Disc (outer boundary) and the Optic Cup (inner core). By leveraging **Connected Component Labeling**, the model ensures that only the most significant anatomical structures are segmented, ignoring noise and small bright lesions (exudates).

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
pip install opencv-python numpy pandas matplotlib

```

### 2. Basic Usage

```python
import cv2
from segmentation_main import run_experiment, get_drishti_paths

# Initialize dataset paths
samples = get_drishti_paths("path/to/Drishti-GS", mode="Testing")

# Run optimized segmentation (90th percentile for Disc, 75th for Cup)
results = run_experiment(samples, disc_perc=90, cup_perc=75, save_images=True)
print(f"Average Disc Dice: {results[0]}")

```

---

## 📊 Results Summary

The following results represent the "Optimized" configuration after multiple intensity experiments.

| Structure | Dice Coefficient | Pixel Accuracy |
| --- | --- | --- |
| **Optic Disc** | 0.9122	 | 0.9855 |
| **Optic Cup** | 0.7845 | 0.9710 |
| **Background** | 0.9910 | 0.9940 |

*Average Dice (Disc + Cup): 0.848 | Connectivity: 8-neighbor*

---

## 🗂️ Project Structure

```text
├── Drishti-GS/              # Dataset (Images and Ground Truth Masks)
├── image_segmentation.pdf   # Project report
├── output_samples/          # Generated segmentation results
├── segmentation_main.py     # Core logic and experiment runner
├── README.md                # Project documentation
└── requirements.txt         # List of dependencies

```

---

## 🔍 Methodology

The pipeline is divided into three critical phases:

1. **Field of View (FOV) Extraction**: A circular mask is applied to the image center to isolate the retina and remove black corner noise.
2. **V-set Thresholding**: Adaptive intensity thresholds are calculated using percentiles of the gray-level histogram.
3. **Connected Component Analysis**: An 8-connectivity algorithm labels all bright regions; the largest component is selected as the Optic Disc to ensure anatomical consistency.

---

## 👤 Author

**[Aleeza Rizwan](linkedin.com/in/aleeza-rizwan/)** Digital Image Processing

---

> [!NOTE]
> Developed for Academic Research. This tool is a decision-support prototype and is not intended for direct clinical diagnosis without practitioner oversight.
