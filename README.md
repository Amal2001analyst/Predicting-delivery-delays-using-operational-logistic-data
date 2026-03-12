# Predicting Delivery Delays Using Operational Logistics Data
**Capstone Project - Group 7**

## 📌 Project Overview
This project focuses on predicting last-mile delivery performance using anonymized operational logistics data provided by Prozo Technologies Pvt. [cite_start]Ltd.[cite: 576]. [cite_start]The primary goal is to develop a machine learning model that classifies customer orders as "on-time" or "delayed" at or shortly after order creation[cite: 577]. [cite_start]This enables proactive operational interventions, shifting logistics from a reactive to a predictive model[cite: 72, 577].

## 🏢 Business Problem
[cite_start]E-commerce logistics companies handle millions of shipments daily, where unpredictable delays lead to customer dissatisfaction, increased support costs, and loss of brand trust[cite: 64, 66]. [cite_start]Identifying delays only after they occur leaves no room for proactive action[cite: 67]. [cite_start]Our solution flags high-risk shipments early for prioritization or courier reassignment[cite: 101].

## 📊 Dataset Description
[cite_start]The dataset includes over **250,000 shipment records**[cite: 74, 117].
- [cite_start]**Target Variable:** `delayed` (Binary: 0 for On-time, 1 for Delayed)[cite: 577].
- [cite_start]**Key Features:** Shipment weight, dimensions, courier partner, pickup/delivery locations, geographic zone, and delivery commitment windows (Min/Max TAT)[cite: 127].
- [cite_start]**Engineered Features:** Created advanced features like Shipment Volume, Chargeable Weight Proxy, Courier-Zone Interaction, and Promise Window tightness[cite: 96, 124].

## 🛠️ Methodology & Algorithms
We followed an end-to-end data science pipeline:
1. [cite_start]**EDA & Data Cleaning:** Addressed class imbalance (~20% delays) and high-cardinality variables[cite: 20, 120].
2. [cite_start]**Feature Engineering:** Derived operational features capturing real-world logistics constraints[cite: 93].
3. [cite_start]**Modeling:** Evaluated multiple algorithms, including Logistic Regression (baseline), Random Forest, XGBoost, and CatBoost[cite: 22, 23].
4. [cite_start]**Optimization:** Performed hyperparameter tuning using `RandomizedSearchCV` and `GridSearchCV`[cite: 34, 37].

## 🏆 Key Results
* [cite_start]**Best Model:** **CatBoost** achieved the highest **ROC AUC of 0.808**[cite: 31, 45].
* [cite_start]**Critical Metrics:** Prioritized **Recall** for delay prediction, catching 9% more actual delays in the baseline CatBoost model[cite: 29, 30].
* [cite_start]**Findings:** Geography and specific courier-zone combinations are the primary drivers of delays, with "METROS" zones consistently showing higher delay rates (35-40%)[cite: 47, 200].

## 🚀 Deployment Recommendation
[cite_start]For real-world deployment, we recommend using the **Phase 3 CatBoost weights** with a lowered classification threshold (< 0.5) to maximize recall without retraining[cite: 49].

---
[cite_start]**Team Members:** Roshni (Team Lead), Sashank Bhargava, Amal Krishna A, Sri Shewatha K[cite: 59, 560].
[cite_start]**Mentor:** Gaurav Chauhan[cite: 57].
[cite_start]**Batch:** PGP in Data Science with Specialization in GenAI-Online Apr25[cite: 560].
