# 👁️ Face Recognition with Machine Learning & Flask Web App

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

> Sistem klasifikasi wajah berbasis Machine Learning menggunakan **SVM**, **Random Forest**, dan **KNN** dengan antarmuka web interaktif menggunakan **Flask**.

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem pengenalan wajah yang mampu mengklasifikasikan kondisi mata seseorang (terbuka / berkacamata hitam) berdasarkan gambar wajah. Dataset berupa gambar `.png` dengan nama file yang mengandung informasi metadata seperti **userid**, **pose**, **ekspresi**, dan **kondisi mata**.

Pipeline machine learning meliputi:
- Preprocessing gambar (grayscale, resize, normalisasi)
- Ekstraksi fitur menjadi vektor 1D
- Training 3 model ML dengan **GridSearchCV** (hyperparameter tuning)
- Evaluasi dan visualisasi hasil (Confusion Matrix)
- Deployment model terbaik via **Flask Web App**

---

## 🗂️ Struktur Folder

```
face-recognition-ml/
│
├── machine learning/                        # Folder training model ML
│   ├── .ipynb_checkpoints/                  # Checkpoint otomatis Jupyter
│   │   ├── TUBES ML (tanpa comment)-c...    # Checkpoint notebook (tanpa komentar)
│   │   └── TUBES ML-checkpoint.ipynb        # Checkpoint notebook utama
│   ├── faces-png/                           # Dataset gambar wajah
│   │   └── <userid>/
│   │       └── <userid>_<pose>_<expression>_<eyes>_<scale>.png
│   ├── TUBES ML (tanpa comment).ipynb       # Notebook bersih (tanpa komentar)
│   ├── TUBES ML.ipynb                       # Notebook utama dengan penjelasan
│   └── svm_model.pkl                        # Model SVM hasil training
│
├── web/                                     # Folder Flask Web App
│   ├── static/
│   │   ├── uploads/                         # Gambar hasil upload user
│   │   └── style.css                        # Styling halaman web
│   ├── templates/
│   │   └── index.html                       # Halaman utama (UI prediksi)
│   ├── app.py                               # Backend Flask
│   ├── requirements.txt                     # Daftar dependensi Python
│   └── svm_model.pkl                        # Salinan model untuk web app
│
└── README.md                                # Dokumentasi proyek
```

---

## 🧠 Format Nama File Dataset

```
123_front_smile_open_2.png    → 5 bagian (ada scale)
456_left_neutral_closed.png   → 4 bagian (scale default = 1)
```

| Bagian | Contoh | Keterangan |
|--------|--------|------------|
| userid | `123` | ID pengguna |
| pose | `front` | Posisi wajah |
| expression | `smile` | Ekspresi wajah |
| eyes | `open` / `sunglasses` | Kondisi mata (label target) |
| scale | `2` | Skala gambar (opsional) |

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/face-recognition-ml.git
cd face-recognition-ml
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Notebook (Training)

Buka folder `machine learning/` lalu jalankan `TUBES ML.ipynb` dengan Jupyter. Jalankan seluruh sel untuk melatih model dan menyimpan `svm_model.pkl`.

### 4. Jalankan Web App

Masuk ke folder `web/` terlebih dahulu:

```bash
cd web
python app.py
```

Buka browser dan akses: `http://127.0.0.1:5000`

> 💡 **Tip:** Tekan `Ctrl + Click` pada link yang muncul di terminal untuk membuka langsung.

---

## 🔬 Pipeline Machine Learning

### Preprocessing Gambar

```python
def load_and_preprocess_image(path, size=(64, 60)):
    img = Image.open(path).convert('L')   # Grayscale
    img = img.resize(size)                # Resize 64×60
    img_array = np.array(img) / 255.0    # Normalisasi 0–1
    return img_array.flatten()            # Flatten ke vektor 1D (3840,)
```

### Model yang Digunakan

| Model | Library | Hyperparameter Tuning |
|-------|---------|----------------------|
| **Support Vector Machine (SVM)** | `sklearn.svm.SVC` | C, kernel, gamma |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | GridSearchCV |
| **K-Nearest Neighbors (KNN)** | `sklearn.neighbors.KNeighborsClassifier` | GridSearchCV |

- **Cross-validation:** 5-fold
- **Evaluasi:** Accuracy Score, Classification Report, Confusion Matrix

---

## 🌐 Cara Kerja Web App (Flask)

1. User mengupload gambar wajah melalui form
2. Flask menyimpan gambar ke `static/uploads/`
3. Gambar di-preprocess (grayscale → resize → flatten)
4. Model SVM memprediksi label (kondisi mata)
5. Hasil prediksi dan gambar ditampilkan di halaman web

```
User Upload Gambar
       ↓
  Flask (app.py)
       ↓
  Preprocessing
       ↓
  SVM Prediction
       ↓
  Tampilkan Hasil di index.html
```

---

## 📊 Hasil & Evaluasi

- Evaluasi menggunakan **Accuracy Score**, **Classification Report**, dan **Confusion Matrix**
- Confusion Matrix divisualisasikan menggunakan `ConfusionMatrixDisplay` dengan colormap `Blues`
- Waktu eksekusi training dicatat menggunakan `time.time()`

---

## 📦 Requirements

```
flask
scikit-learn
numpy
pandas
Pillow
matplotlib
joblib
```

Install semua sekaligus:

```bash
pip install -r requirements.txt
```

---
