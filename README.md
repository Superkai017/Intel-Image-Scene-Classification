# Intel-Image-Scene-Classification
# Intel Image Scene Classification

A deep learning project that classifies natural scene images into six categories using Convolutional Neural Networks (CNNs) and transfer learning.

## Overview

This project trains an image classifier on the **Intel Image Classification** dataset, originally published by Intel for an image classification challenge. The goal is to predict the scene shown in a given image.

**Classes:**

| Label | Description |
|-------|-------------|
| `buildings` | Urban and architectural scenes |
| `forest` | Woodland and trees |
| `glacier` | Ice and snow formations |
| `mountain` | Mountain landscapes |
| `sea` | Ocean and coastal scenes |
| `street` | Roads and street views |

## Dataset

- **Source:** [Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Size:** ~25,000 images
- **Image resolution:** 150 × 150 pixels, RGB
- **Splits:**
  - `seg_train` – ~14,000 images (training)
  - `seg_test` – ~3,000 images (validation / testing)
  - `seg_pred` – ~7,000 unlabeled images (prediction)

Expected directory structure:

```
data/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── seg_test/
│   └── seg_test/
│       └── (same six class folders)
└── seg_pred/
    └── seg_pred/
        └── (unlabeled images)
```

## Project Structure

```
.
├── data/                   # Dataset (not tracked in git)
├── notebooks/
│   └── intel_scene_classification.ipynb
├── src/
│   ├── dataset.py          # Data loading and augmentation
│   ├── model.py            # Model architectures
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Metrics and confusion matrix
│   └── predict.py          # Inference on new images
├── models/                 # Saved weights
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/intel-image-scene-classification.git
cd intel-image-scene-classification

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Typical requirements:**

```
tensorflow>=2.12
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
```

(Swap TensorFlow for `torch` and `torchvision` if you use PyTorch.)

## Download the Data

Using the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip -d data/
```

## Usage

**Train a model:**

```bash
python src/train.py --epochs 30 --batch-size 32 --model efficientnet
```

**Evaluate on the test set:**

```bash
python src/evaluate.py --weights models/best_model.h5
```

**Predict a single image:**

```bash
python src/predict.py --image path/to/image.jpg --weights models/best_model.h5
```

Or open the notebook and run all cells:

```bash
jupyter notebook notebooks/intel_scene_classification.ipynb
```

## Methodology

1. **Preprocessing** – Rescale pixel values to [0, 1] and resize images consistently.
2. **Data augmentation** – Random flips, rotations, zooms, and brightness shifts to reduce overfitting.
3. **Models**
   - A custom CNN baseline (Conv → BatchNorm → MaxPool blocks with dropout)
   - Transfer learning with pretrained backbones (e.g., VGG16, ResNet50, MobileNetV2, EfficientNet)
4. **Training** – Adam optimizer, categorical cross-entropy loss, learning-rate scheduling, and early stopping.
5. **Evaluation** – Accuracy, precision, recall, F1-score, and a confusion matrix.

## Results

Fill in with your own numbers after training.

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| Custom CNN | – | – |
| VGG16 (fine-tuned) | – | – |
| ResNet50 (fine-tuned) | – | – |
| EfficientNet (fine-tuned) | – | – |

**Common observations:** Models often confuse `glacier` with `mountain`, and `buildings` with `street`, because these scenes share visual features.

## Future Improvements

- Experiment with Vision Transformers (ViT)
- Add Grad-CAM visualizations to interpret predictions
- Apply test-time augmentation
- Deploy as a web app with Streamlit or Gradio
- Export to ONNX or TensorFlow Lite for edge deployment

## Acknowledgments

- Dataset provided by Intel and shared on Kaggle by [Puneet Bansal](https://www.kaggle.com/puneet6060).
- Pretrained model weights from ImageNet.

## License

This project is released under the MIT License. See `LICENSE` for details. Note that the dataset is subject to its own license terms on Kaggle.

## Contact

Your Name – your.email@example.com
Project link: https://github.com/<your-username>/intel-image-scene-classification
