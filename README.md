---


# Face Recognition System (SVM + PCA)

This repository contains a **Face Recognition System** implemented using **Supervised Machine Learning** techniques.  
The system uses **Principal Component Analysis (PCA)** for feature reduction and **Support Vector Machine (SVM)** for face classification.

The project is **fully dynamic**, allowing users to add new people, recreate the dataset, retrain the model, and test recognition results.
---

## 📁 Project Structure

```

├── .ipynb_checkpoints/
│
├── train_data_2/                     # Training face images
├── test_data/                        # Testing face images
│
├── data.npy                          # PCA-transformed face features
├── target.npy                        # Class labels
│
├── haarcascade_frontalface_default.xml  # Face detection model
│
├── fscrRecodDatasetMaking.ipynb      # Dataset creation & feature extraction
├── trainModel(SVM).ipynb             # PCA + SVM training
├── testModel.ipynb                   # Model testing
│
├── SVM-FaceRecognition.sav            # Trained SVM model
└── README.md

```

---

## 🧠 Machine Learning Approach

### 🔹 Supervised Learning
- The system follows a **supervised learning approach**.
- Each face image is labeled with the corresponding **person name**.
- Labels are used to train the classifier.

### 🔹 PCA (Principal Component Analysis)
- PCA is applied to:
  - Reduce dimensionality
  - Remove noise
  - Improve training speed
- Converts face images into compact feature vectors.

### 🔹 SVM (Support Vector Machine)
- SVM is used as the final classifier.
- Learns decision boundaries between different individuals.
- Performs well on high-dimensional face data.

---

## ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Jupyter Notebook

---

## 📊 Dataset Structure

### Training Data
```

train_data_2/
├── saman_kumara/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── kasun_perera/
└── nimal_silva/

```

### Testing Data
```

test_data/
├── saman_kumara/
├── kasun_perera/

```

Each folder represents **one class (one person)**.

---

## 🔁 Train Your Own Model (Dynamic Dataset)

This system supports **dynamic training**.

### ✅ Step 1: Add New Person
1. Navigate to `train_data_2/`
2. Create a new folder with the person’s name  
   Example:
```

train_data_2/saman_kumara/

````
3. Add multiple face images of that person.

---

### ✅ Step 2: Create Dataset
Run:
```text
fscrRecodDatasetMaking.ipynb
````

This will:

* Detect faces
* Apply PCA
* Generate:

  * `data.npy`
  * `target.npy`

---

### ✅ Step 3: Train the Model

1. Open:

   ```text
   trainModel(SVM).ipynb
   ```
2. Ensure the new person’s name is included in the label list.
3. Train the SVM classifier.

Output:

```
SVM-FaceRecognition.sav
```

---

### ✅ Step 4: Test the Model

1. Add images to:

   ```
   test_data/saman_kumara/
   ```
2. Run:

   ```text
   testModel.ipynb
   ```

The model will predict the person’s identity.

---

## 📈 Model Summary

| Component                | Description             |
| ------------------------ | ----------------------- |
| Learning Type            | Supervised Learning     |
| Dimensionality Reduction | PCA                     |
| Classifier               | SVM                     |
| Output                   | Person Name             |
| Model File               | SVM-FaceRecognition.sav |

---

## 🔐 Ethical Notice

This project is intended **only for academic and educational purposes**.
Avoid using it for unauthorized surveillance or privacy-sensitive applications.

---

## ✍️ Author

**Madhuka Malshan**
Software Engineering Student
National Institute of Business Management (NIBM), Sri Lanka



---
