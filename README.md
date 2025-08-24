# 📌 Project Overview
This project focuses on Anomaly Detection in the Tennessee Eastman Process (TEP), a well-known industrial benchmark used for process control and fault detection research.

We implement multiple machine learning and deep learning models to detect anomalies in TEP sensor data, with options for fast approximate detection or accurate but computationally heavier detection.

The pipeline is modular and extensible, allowing you to:

   - Run anomaly detection using Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, PCA, and Autoencoder (TensorFlow/Keras).
    
   - Select between Fast Mode (optimized for speed, still preserves feature importance length) and Accurate Mode (full evaluation across models).
    
   - Extract Top 7 contributing features per data entry, explaining anomalies.
    
   - Save and visualize model results for further analysis.

# ⚙️ Installation & Setup
1️⃣ Clone the Repository
```
git clone https://github.com/kunal2026/HoneyWell_TEP.git
```
```
cd HoneyWell_TEP
```

2️⃣ Create Virtual Environment
```
python -m venv venv
```
Linux / macOS
```
source venv/bin/activate
```
Windows PowerShell
```
venv\Scripts\activate
```
3️⃣ Install Dependencies
```
pip install -r web/requirements.txt
```
(for AutoEncoders)
```
pip install tensorflow
```
# 🚀 Usage
The main script is models/run_algorithms.py.
It supports:

Modes: fast (speed) or accurate (full evaluation).

Models: choose one (isoforest, lof, svm, autoencoder, pca) or all.

Run:
```
python web/app.py
```

After that select:

    1. File
    2. Model
    3. Mode

    
Output:
   - Top 7 anomaly features per entry as columns.
   - Anomaly scores + predictions saved in results.
   - This file can be downloaded.

# 📊 Models Implemented
 - Isolation Forest (Sklearn)

 - Local Outlier Factor (LOF)

 - KNN

 - Mahalanobis

 - One-Class SVM

 - Principal Component Analysis (PCA-based anomaly detection)

 - Deep Autoencoder (TensorFlow/Keras)

# 🔧 Configuration
You can configure:

Dataset path (data/ directory)

Model parameters (edit config.json or modify arguments)

Output - Downloadable sheet

# 🧪 Example Workflow
Prepare the dataset in data/.

Run anomaly detection:
```
python web/app.py
```
Download the sheet from the given option.

anomaly_scores.csv → anomaly scores per sample

top_features.csv → Top 7 contributing features per entry


# 📚 References
_[1] A. Zimek and P. Filzmoser, "There and back again: Outlier detection between statistical reasoning and data mining algorithms," Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, vol. 3, no. 1, pp. 1–17, Jan. 2013._

_[2] E. Gujral, "Survey: Anomaly Detection Methods," UCR MADLab, 2023._

_[3] S. Pimentel, D. Clifton, L. Clifton, and L. Tarassenko, "A review of novelty detection," Signal Processing, vol. 99, pp. 215–249, Apr. 2014._

_[4] P. Rousseeuw and A. Leroy, Robust Regression and Outlier Detection, Wiley, 1987._ 

_[5] A. Chatterjee and B. S. Ahmed, "IoT anomaly detection methods and applications: A survey," arXiv, vol. 2207, p. 9092, Jul. 2022._

_[6] Scikit-learn documentation: https://scikit-learn.org/_

_[7] TensorFlow documentation: https://www.tensorflow.org/_
