#  Customer Churn Prediction

A **Machine Learning web application** built using **Flask** that predicts whether a customer is likely to churn or not.  
The project demonstrates a complete end-to-end ML workflow — from **data preprocessing and model training** to **web deployment** using Flask and Render.

---

##  Features

- Interactive web interface built with **Flask**
- Real-time customer churn prediction
- Input data scaled with **StandardScaler**
- Trained **Random Forest Classifier** for prediction
- Ready for deployment on **Render**

---

##  Model Overview

The model was trained using a telecom customer churn dataset with the following steps:

1. **Data Cleaning** – Removed irrelevant columns (`Surname`, `Gender`, `RowNumber`, `CustomerId`)  
2. **Encoding** – Converted `Geography` column into dummy variables  
3. **Scaling** – Applied `StandardScaler` to numerical features  
4. **Modeling** – Trained using `RandomForestClassifier`  
5. **Evaluation** – Achieved accuracy of **~85–90%**

---

##  Project Structure
customer-churn-prediction/
│
├── templates/
│   └── index.html             # Frontend HTML form
│
├── static/
│   └── style.css              # Styling for frontend
│
├── customer_churn_model.pkl   # Trained Random Forest model
├── scaler.pkl                 # StandardScaler used during training
├── app.py                     # Flask backend
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

---

##  Technologies Used

- **Python 3.10+**
- **Flask**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **HTML / CSS**
- **Render** for deployment

---



Author
Nilay Shahane
Agentic AI , Machine Learning & Full Stack Developer
📧 nilayshahane@gmail.com


---



