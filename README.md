# 📓 Deepfake Detection — Model Training Notebooks

![Static Badge](https://img.shields.io/badge/Deepfake%20Training-Notebooks-blueviolet?style=for-the-badge\&logo=jupyter\&logoColor=orange)
![Static Badge](https://img.shields.io/badge/Python-3.12-green?style=flat\&logo=python)
![Static Badge](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat\&logo=tensorflow)
![Static Badge](https://img.shields.io/badge/PyTorch-2.2.2-red?style=flat\&logo=pytorch)
![Static Badge](https://img.shields.io/badge/Kaggle-Dataset-blue?style=flat\&logo=kaggle)
![Static Badge](https://img.shields.io/badge/Colab-Compatible-yellow?style=flat\&logo=googlecolab)



## 📌 About the Project

This repository contains **Jupyter notebooks** used for training and evaluating various deep learning models for **Deepfake Detection**.

The goal is to classify whether a given video or image is **real or fake** (deepfake) using CNN-based architectures, hybrid models, and transformers. These models are later used in a production-ready microservices-based detection system.

## 🧠 What is Deepfake?

Deepfake technology uses AI (especially GANs) to create fake but highly realistic audio/video content by manipulating a person’s face or voice. While it's an exciting tech frontier, it poses serious risks to trust, privacy, and security.



## 🏗️ Model Architectures

This repository includes implementations and training pipelines for the following model variants:

### 🌀 CNN + ViT (Vision Transformer)

* Combines **Convolutional Neural Networks** for spatial feature extraction with **ViT** for long-range attention modeling.
* Used **ResNet50** or **EfficientNet** as CNN backbone.
* ViT encodes patches extracted from feature maps.

> 🔬 **Strengths**: Global context awareness + strong spatial features
> 📈 **Use Case**: Works well on both face images and video frames.

---

### 📽 CNN + LSTM

* CNN extracts features from video frames.
* **LSTM** (Long Short-Term Memory) models temporal relationships.

> ⏱️ **Strengths**: Great for frame-sequence modeling
> 🎞️ **Use Case**: Best suited for temporal deepfake classification.

---

### 🎥 CNN + GRU

* GRU is a lighter version of LSTM.
* Faster and more efficient for long video sequences.

> 🚀 **Strengths**: Faster training than LSTM, good for medium-sized datasets
> 💡 **Use Case**: Used in rapid prototyping and low-latency use cases.



---

## 🧪 Training Details

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Input Size        | 224x224 (for all models)        |
| Batch Size        | 16–64                           |
| Optimizer         | Adam / AdamW                    |
| Loss Function     | BinaryCrossEntropy              |
| Dataset           | Custom (Kaggle + FaceForensics) |
| Training Platform | Google Colab + Kaggle Notebook    |

---

## 🛠 Tools Used

| Purpose       | Tool/Library        |
| ------------- | ------------------- |
| Deep Learning | TensorFlow, PyTorch |
| Vision Model  | Transformers (ViT)  |
| Visualization | Matplotlib, Seaborn |
| Video/Frame   | OpenCV              |
| Data Handling | Pandas, NumPy       |

<!-- ---

## 📈 Results Summary

| Model           | Accuracy  | F1 Score | ROC-AUC  |
| --------------- | --------- | -------- | -------- |
| CNN + ViT       | 93.2%     | 0.91     | 0.95     |
| CNN + LSTM      | 91.7%     | 0.89     | 0.94     |
| CNN + GRU       | 90.4%     | 0.87     | 0.92     |
| **Final Model** | **95.1%** | **0.93** | **0.97** |

---

## 📎 Try It Yourself

Open notebooks in Google Colab:

* ▶️ [CNN + ViT Notebook](https://colab.research.google.com/github/your-username/deepfake-training-notebooks/blob/main/cnn_vit_training.ipynb)
* ▶️ [CNN + LSTM Notebook](https://colab.research.google.com/github/your-username/deepfake-training-notebooks/blob/main/cnn_lstm_training.ipynb)
* ▶️ [CNN + GRU Notebook](https://colab.research.google.com/github/your-username/deepfake-training-notebooks/blob/main/cnn_gru_training.ipynb)
* 🏁 [Final Model Notebook](https://colab.research.google.com/github/your-username/deepfake-training-notebooks/blob/main/final_model_training.ipynb)

--- -->

## 📌 Related Projects

* 🔗 **[Microservices Deployment Repo](https://github.com/KarnamShyam1947/deepfake-detection-backend)**

> End-to-end microservices-based system for deepfake detection


