# Multimedia Authentication Using Multi-Feature Fusion

##  Project Idea — Current Direction

Our project aims to detect whether an image has been **authentic (real)** or **manipulated/forged** by combining information from different feature domains.

###  Main Idea

We are currently exploring **Spatial + Frequency Feature Fusion** for image forgery/deepfake detection.

An image contains useful information in:

* **Spatial domain** → visual patterns, textures, edges, facial details, etc.
* **Frequency domain** → high-frequency artifacts and manipulation traces that may not be obvious in the normal RGB image.

Instead of relying on only one type of information, we will extract features from **both domains** and combine them before classification.

###  Proposed Pipeline

```text
                    Input Image
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        RGB Image              Frequency Image
              │                (Haar DWT)
              │                     │
              ▼                     ▼
          ResNet18              ResNet18
              │                     │
              ▼                     ▼
      Spatial Features       Frequency Features
              │                     │
              └──────────┬──────────┘
                         ▼
                   Feature Fusion
                         │
                         ▼
                    Classifier
                         │
                    ┌────┴────┐
                    ▼         ▼
                  Real      Fake
```

###  Initial Method

We will start with a simple and reproducible approach:

* **Dataset:** FaceForensics++
* **Task:** Real vs Manipulated image/frame classification
* **Spatial backbone:** ResNet18
* **Frequency transform:** Haar Discrete Wavelet Transform (DWT)
* **Fusion:** Concatenation of spatial + frequency features
* **Classifier:** Fully connected layers
* **Loss:** Cross Entropy

###  Experiments

We will compare three progressively stronger models:

**Experiment 1 — Spatial Baseline**

```text
RGB → ResNet18 → Classifier → Real/Fake
```

**Experiment 2 — Frequency Baseline**

```text
RGB → Haar DWT → ResNet18 → Classifier → Real/Fake
```

**Experiment 3 — Multi-Feature Fusion**

```text
RGB → ResNet18 ───────┐
                      ├→ Feature Fusion → Classifier → Real/Fake
DWT → ResNet18 ───────┘
```

This will allow us to determine whether combining spatial and frequency information actually improves detection.

###  Evaluation

We will evaluate using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

Later, if the basic system works, we may investigate a small improvement such as:

* Attention/adaptive feature fusion
* Different frequency transforms
* Lightweight preprocessing
* Robustness against JPEG compression, blur, resizing, etc.

###  Important

This is our **current project direction**, based on the literature we have explored so far.

We have **not finalized the exact methodology/paper to reproduce yet**.

Our immediate goal is:

> **Explore the dataset → reproduce simple spatial and frequency baselines → implement feature fusion → compare results → then decide on improvements based on existing research.**

###  Collaboration

We will use:

* **GitHub** → code, notebooks, documentation
* **Google Colab** → development and GPU training
* **Kaggle/Google Drive** → dataset storage

The dataset and large model checkpoints should **not** be uploaded to GitHub.
