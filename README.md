# Predicting Delivery Delays Using Operational Logistics Data
**Capstone Project – Group 7**

---

## 📌 Project Overview
This project focuses on predicting last-mile delivery delays in e-commerce logistics using machine learning techniques. The objective is to identify orders that are likely to be delivered late at an early stage in the order lifecycle, enabling logistics teams to take proactive operational actions.

Using anonymized operational logistics data provided by **Prozo Technologies Pvt. Ltd.**, the project develops a classification model that predicts whether an order will be delivered **on time** or **delayed**. This helps shift logistics operations from a reactive approach to a predictive and data-driven decision-making system.

---

## 🏢 Business Problem
In modern e-commerce logistics, delivery punctuality is critical for customer satisfaction, repeat purchases, and brand reputation.

However, delays are often identified **only after they occur**, leaving operations teams with limited ability to intervene. Logistics networks involve multiple warehouses, courier partners, and delivery zones, making delay prediction complex.

This project aims to:
- Predict delivery delays early in the order lifecycle
- Identify operational factors contributing to delays
- Enable proactive actions such as courier reassignment or order prioritization

---

## 📊 Dataset Description
The dataset contains **250,000+ anonymized shipment records** shared by Prozo for academic analysis.

### Target Variable
- **`delayed`**  
  - `0` → On-time delivery  
  - `1` → Delayed delivery

### Key Features
- Shipment weight and dimensions  
- Courier partner information  
- Pickup and delivery locations  
- Geographic delivery zones  
- Delivery commitment windows (Min/Max TAT)

### Engineered Features
To improve predictive power, several features were engineered:

- Shipment Volume  
- Chargeable Weight Proxy  
- Courier–Zone Interaction  
- Promise Window Tightness  
- Temporal features (Day of Week, etc.)

---

## ⚙️ Project Workflow
The project follows a structured **data science pipeline**:

1. **Data Understanding & Cleaning**
   - Handling missing values
   - Standardizing timestamp formats
   - Removing inconsistencies

2. **Exploratory Data Analysis (EDA)**
   - Identifying delay patterns
   - Understanding feature relationships
   - Handling class imbalance (~20% delayed orders)

3. **Feature Engineering**
   - Creating logistics-specific operational features
   - Encoding categorical variables

4. **Model Development**
   Multiple classification algorithms were tested:

   - Logistic Regression (Baseline)
   - Decision Tree
   - Random Forest
   - XGBoost
   - CatBoost

5. **Model Optimization**
   - Hyperparameter tuning using:
     - `RandomizedSearchCV`
     - `GridSearchCV`

---

## 🏆 Key Results
- **Best Performing Model:** CatBoost  
- **ROC-AUC Score:** **0.808**

### Key Insights
- Geography and courier-zone combinations strongly influence delays.
- Metropolitan zones show higher delay probabilities (~35–40%).
- Optimizing classification thresholds improves delay detection.

The model prioritizes **Recall**, enabling the system to detect more potential delays before they occur.

---

## 🚀 Deployment Recommendation
For production deployment:

- Use the **Phase 3 CatBoost model**
- Apply a **lower classification threshold (< 0.5)** to increase delay detection
- Integrate the model into an **operations dashboard** to flag high-risk shipments at order creation.

This would allow logistics teams to take preventive actions such as courier reassignment or order prioritization.

---

