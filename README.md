
# 🖥️ PC Components Classification AI

### Deep Learning | Computer Vision | Transfer Learning | Python

An AI-powered image classification project that recognizes different computer hardware components using **Deep Learning and Computer Vision**.

The project uses **TensorFlow, Keras, and MobileNetV2** with Transfer Learning to classify computer components from images.

## 🚀 Project Overview

The goal of this project is to develop an intelligent image recognition system capable of identifying computer hardware components.

The model learns visual features from images and predicts the corresponding hardware category.

### Key Features

- Computer hardware image classification
- Transfer Learning using MobileNetV2
- Image preprocessing and augmentation
- Multi-class classification
- Model training and evaluation
- Prediction on new images
- Saved model for future deployment
- Webcam integration for real-time predictions

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning framework |
| Keras | Neural network development |
| MobileNetV2 | Pre-trained CNN architecture |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Jupyter Notebook | Model development |
| OpenCV | Webcam image processing |

## 🏗️ Model Architecture

The project uses **MobileNetV2**, a pre-trained Convolutional Neural Network, as a feature extractor.

Transfer Learning allows the model to reuse previously learned visual features and adapt them to computer hardware classification.

### Workflow

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
MobileNetV2
(Transfer Learning)
     |
     v
Feature Extraction
     |
     v
Classification Layers
     |
     v
Softmax Activation
     |
     v
Predicted PC Component
```

## 🖥️ Hardware Categories

The dataset contains images of different computer components, including:

- Cables
- Computer Cases
- CPUs
- GPUs
- HDDs
- Headsets
- Keyboards
- Microphones
- Monitors
- Motherboards
- Mice
- RAM

## 📊 Model Performance

The trained model achieved the following results on the test dataset:

| Metric | Result |
|---|---|
| Test Accuracy | **73.37%** |
| Test Loss | **0.9510** |
| Model | MobileNetV2 |
| Learning Approach | Transfer Learning |
| Task | Multi-class Image Classification |

### 🎯 Test Accuracy: 73.37%

The results demonstrate the model's ability to recognize multiple computer hardware categories from previously unseen images.

Further improvements may be achieved through fine-tuning, additional training data, and hyperparameter optimization.

## 📂 Project Structure

```text
PC-Components-Classification-AI/
│
├── newpredictAI.ipynb
├── camera.py
├── README.md
│
├── model/
│   └── trained_model.keras
│
└── dataset/
    ├── train/
    ├── val/
    └── test/
```

*Note: The structure above is a suggested repository layout. Update the model filename and dataset paths to match your local project.*

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PC-Components-Classification-AI.git
cd PC-Components-Classification-AI
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib opencv-python pillow
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook newpredictAI.ipynb
```

### 4. Run webcam recognition

After configuring the saved model path in `camera.py`:

```bash
python camera.py
```

## 🔬 Machine Learning Pipeline

1. Collect computer hardware images.
2. Organize images into categories.
3. Split the dataset into training, validation, and testing sets.
4. Preprocess and resize images.
5. Apply Transfer Learning with MobileNetV2.
6. Train the classification layers.
7. Evaluate the model on unseen test images.
8. Save the trained model.
9. Integrate the model with a webcam application.

## 🔮 Future Improvements

- Improve classification accuracy
- Fine-tune additional MobileNetV2 layers
- Expand the training dataset
- Improve predictions under different lighting conditions
- Build a Flask web application
- Deploy the model as an AI-powered application
- Add real-time confidence score visualization

## 👨‍💻 About the Developer

I am an aspiring **Machine Learning / AI Engineer** passionate about Python, Deep Learning, and Computer Vision.

I enjoy building practical AI applications, experimenting with neural networks, and turning machine learning models into real-world projects.

### Areas of Interest

- Machine Learning
- Deep Learning
- Computer Vision
- Artificial Intelligence
- Python Development
- Data Science

## ⭐ Support

If you find this project interesting, consider giving the repository a star!

**Built with Python, TensorFlow, and a passion for Artificial Intelligence.**
