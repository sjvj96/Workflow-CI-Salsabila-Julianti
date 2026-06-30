# Workflow CI - Adult Income Classification

## Deskripsi
Project ini merupakan implementasi workflow Machine Learning menggunakan MLflow dan GitHub Actions (CI) untuk melakukan klasifikasi pendapatan (Adult Income Classification).

Model yang digunakan adalah Logistic Regression dengan dataset Adult Income yang telah melalui tahap preprocessing.

---

## Struktur Project

```
MLProject/
│
├── adult_preprocessed.csv
├── modelling.py
├── requirements.txt
├── python_env.yaml
├── conda.yaml
└── MLproject

.github/
└── workflows/
    └── ci.yml
```

---

## Library

- Python 3.12
- MLflow 2.19.0
- Pandas
- NumPy
- Scikit-learn

---

## Cara Menjalankan

### Install dependency

```bash
pip install -r requirements.txt
```

### Jalankan MLflow Project

```bash
mlflow run MLProject --env-manager local
```

atau

```bash
cd MLProject
python modelling.py
```

---

## Hasil

Project ini melakukan:

- Load dataset
- Split data train dan test
- Training Logistic Regression
- Evaluasi model
- Logging parameter, metric, dan model menggunakan MLflow

---

## Continuous Integration

Workflow GitHub Actions akan berjalan secara otomatis setiap kali terjadi:

- Push ke branch `main`
- Pull Request ke branch `main`

File workflow berada pada:

```
.github/workflows/ci.yml
```

---

## Author

Salsabila Julianti