# Glaucoma Screening via Retinal Image Segmentation

## 📌 Overview

This project implements a medical image processing pipeline to segment the **Optic Disc (OD)** and **Optic Cup (OC)** from retinal fundus images. These structures are critical in calculating the **Cup-to-Disc Ratio (CDR)**, a primary clinical metric used by ophthalmologists to detect **Glaucoma**.

The solution utilizes **Connected Component Labeling (CCL)** and intelligent **V-set (Intensity Set)** selection to isolate anatomical structures without the need for complex deep learning models.

---

## 🚀 Key Features

* **Intelligent V-set Selection:** Leverages percentile-based intensity thresholding to adapt to varying image brightness.
* **8-Connectivity Analysis:** Uses Connected Component Labeling to filter out vascular noise and isolate the largest anatomical structures.
* **Automated FOV Masking:** Implements a circular Field of View (FOV) mask to ignore non-retinal artifacts in the fundus photographs.
* **Performance Metrics:** Evaluates accuracy using the **Dice Coefficient** to measure overlap against expert-annotated Ground Truth masks.

---

## 🛠️ Technical Stack

* **Language:** Python
* **Libraries:** * `OpenCV`: Image transformation, thresholding, and CCA.
* `NumPy`: Matrix manipulation and percentile-based intensity analysis.
* `Matplotlib`: Visualization of segmentation results.
* `Pandas`: Data logging for experimental results.



---

## 📂 Project Structure

```bash
├── Drishti-GS/              # Dataset (Images and Ground Truth Masks)
├── image_segmentation.pdf   # Project report
├── output_samples/          # Generated segmentation results
├── segmentation_main.py     # Core logic and experiment runner
├── README.md                # Project documentation
└── requirements.txt         # List of dependencies

```

---

## 🧠 Methodology

The pipeline follows a three-stage process:

### 1. Pre-processing (Field of View)

To prevent background noise from affecting the intensity statistics, a circular mask is applied to the center of the image. This ensures only the retinal area is analyzed.

### 2. Optic Disc Segmentation (Task 1)

A  set is defined to capture the brightest pixels in the image. Using **8-connectivity**, the largest connected component is identified as the Optic Disc, effectively removing small bright spots (exudates) or noise.

### 3. Optic Cup Isolation (Task 2)

The Optic Cup is the brightest region within the Disc. A more restrictive  set is applied specifically to the pixels labeled as "Disc" to isolate the inner cup core.

---

## 📊 Results

Experiments were conducted using varying percentile thresholds to optimize the  set.

| Experiment | Disc Dice | Cup Dice | Accuracy |
| --- | --- | --- | --- |
| Strict | 0.8241 | 0.6512 | 0.9782 |
| Moderate | 0.8854 | 0.7230 | 0.9810 |
| **Optimized** | **0.9122** | **0.7845** | **0.9855** |

---

## 🔧 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/retinal-segmentation.git

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Analysis:**
```bash
python segmentation_main.py

```



---

## 📧 Contact

Developed by [Aleeza Rizwan] – [linkedin.com/in/aleeza-rizwan/]

---
