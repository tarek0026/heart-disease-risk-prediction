# ❤️ Heart Disease Risk Prediction (Machine Learning + Streamlit)

This project is an end-to-end Machine Learning system that predicts the risk of heart disease based on clinical and demographic patient data.  
It demonstrates a complete ML workflow including data preprocessing, model training, evaluation, and deployment using a Streamlit web application.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can help improve treatment decisions and patient outcomes.

In this project, we build a classification model that predicts whether a patient is at risk of heart disease using features such as:

- Age  
- Cholesterol level  
- Resting blood pressure  
- Chest pain type  
- Exercise-induced angina  
- Other medical indicators  

---

## 📊 Dataset

The dataset used is the **UCI Heart Disease Dataset**, which contains real-world medical features and includes missing values.

Target variable:

- `0` → Healthy  
- `1` → Heart Disease Risk  

---

## ⚙️ Machine Learning Pipeline

A full **scikit-learn Pipeline** was implemented to ensure reproducibility and clean preprocessing.

### Numerical Features
- Median Imputation  
- Standard Scaling  

### Categorical Features
- Most Frequent Imputation  
- One-Hot Encoding  

---

## 🤖 Model

The final model used:

- **HistGradientBoostingClassifier**

Hyperparameter tuning was performed using:

- **GridSearchCV**
- **Stratified K-Fold Cross Validation**

---

## 📈 Evaluation

Model performance was evaluated using robust medical metrics:

- **Accuracy**
- **F1-score**
- **ROC-AUC**

Evaluation includes:

✅ Stratified Cross Validation  
✅ Confusion Matrix  
✅ Classification Report  
✅ ROC Curve  
✅ Learning Curve  

Final results achieved approximately:

- **ROC-AUC ≈ 0.87 – 0.90**
- **Accuracy ≈ 0.83 – 0.86**

---

## 🚀 Streamlit Deployment

An interactive Streamlit web app is included where users can input patient data and get a real-time heart disease risk prediction.










## ▶️ How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

2. Train the Model
python src/train.py


This will train the model and save it inside:

models/heart_model.pkl

3. Evaluate the Model
python src/evaluate.py


This prints cross-validation scores and generates evaluation plots.

4. Run the Streamlit App
streamlit run app/streamlit_app.py
## 🧠 Key Learning Outcomes

Building reproducible ML pipelines

Handling missing values properly

Using Stratified Cross Validation

Evaluating healthcare models with ROC-AUC

Saving and deploying models with Streamlit

## ⭐ Future Improvements

Add SHAP explainability for medical interpretation

Deploy the model online using Streamlit Cloud

Experiment with XGBoost and deep learning models

## 👤 Author

Tarek Mohamed El-Batrik
Computer Science Undergraduate @ Nile University
Passionate about Machine Learning and Healthcare AI




## 📂 Project Structure

```text
heart-disease-risk-prediction/
│
├── data/
│   └── raw/
│       └── heart_disease_uci.csv
│
├── models/
│   └── heart_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
