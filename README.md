# Credit Card Customer Churn Prediction using ANN

## 📌 Overview

**Credit Card Customer Churn Prediction using Artificial Neural Network (ANN)** is a machine learning project that predicts whether a credit card customer will churn (leave) or stay. This is a **binary classification** problem solved with **TensorFlow & Keras ANN**.

Churn prediction is crucial in finance because **retaining existing customers is more cost-effective** than acquiring new ones. With this model, businesses can proactively target customers likely to churn with retention strategies.

---

## 🧠 Problem Statement

Given customer data (e.g., credit score, age, balance, geography), the model predicts:

> **Will a customer churn (exit)?**  
> Output: `1` = Yes (Churn), `0` = No (Not churn)

---

## 📂 Repository Structure

Here’s what’s inside this project:

📦Credit-Card-Customer-Churn-Prediction-using-ANN
┣ 📜Churn_Modelling.csv # Dataset file
┣ 📜app.py # Streamlit app for UI
┣ 📜experiments.ipynb # EDA + model building notebook
┣ 📜prediction.ipynb # Sample prediction notebook
┣ 📜model.h5 # Trained ANN model
┣ 📜label_encoder_gender.pkl # Saved encoder for gender
┣ 📜ohe_encoder_geo.pkl # Saved OneHot encoder for geography
┣ 📜scaler.pkl # Saved scaler for input features
┣ 📜requirements.txt # Python dependencies
┗ 📜.gitignore


---

## 📊 Dataset Description

**Churn_Modelling.csv** includes customer demographic and financial attributes. It typically contains:

| Feature | Description |
|---------|-------------|
| CreditScore | Customer’s credit score |
| Geography | Country of customer |
| Gender | Male/Female |
| Age | Customer’s age |
| Tenure | Relationship years with bank |
| Balance | Account balance |
| NumOfProducts | Number of products held |
| HasCrCard | Possession of credit card |
| IsActiveMember | Active customer |
| EstimatedSalary | Estimated income |
| Exited | Target variable (Churn: 1 = Yes, 0 = No) |

> (Dataset format similar to most churn datasets seen in ANN churn projects) :contentReference[oaicite:1]{index=1}

---

## 📌 Project Workflow

### 🧹 1. Data Preprocessing
- Loaded dataset
- Dropped irrelevant columns (e.g., customer ID)
- Label encoded **Gender**
- One-Hot encoded **Geography**
- Applied **Standard Scaling**
- Train-test split

---

### 🤖 2. Model Building (ANN)

Used a sequential ANN model:
- Input layer matching number of features
- 2–3 hidden layers with ReLU activation
- Output layer with Sigmoid for binary classification

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

### 📈 3. Evaluation

Evaluated using:

Accuracy
Confusion Matrix
Classification Report: precision, recall, F1-score
Typical results for churn prediction models are ~85–88% accuracy on test data.

git clone https://github.com/Amankar003/Credit-Card-Customer-Churn-Prediction-using-ANN.git
cd Credit-Card-Customer-Churn-Prediction-using-ANN
pip install -r requirements.txt

| Technology         | Purpose                |
| ------------------ | ---------------------- |
| Python             | Coding & scripting     |
| NumPy, Pandas      | Data manipulation      |
| TensorFlow & Keras | ANN model              |
| Scikit-learn       | Preprocessing, metrics |
| Streamlit          | UI deployment          |
| Jupyter            | Notebooks              |


