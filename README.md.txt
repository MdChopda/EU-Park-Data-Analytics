# 🎢 EU-Park: Data Science Strategy & Analytics
### Optimizing Wait Times, Menu Revenue, and Customer Retention

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Complete-green)
![Focus](https://img.shields.io/badge/Focus-Business%20Intelligence-orange)

## 📋 Executive Summary
This project acts as a data science consulting engagement for "EU-Park," a major theme park. The objective was to analyze operational data to solve three critical business bottlenecks:
1.  **Operational Efficiency:** Predicting attraction wait times to optimize staffing.
2.  **Revenue Maximization:** Identifying menu combinations for "Gold Pass" discount bundles.
3.  **Customer Retention:** Predicting which customers are likely to purchase high-tier Season Passes.

## 🛠️ Technical Architecture
The solution is divided into three analytical modules:

### 1. ⏳ Wait Time Prediction (Operations)
* **Goal:** Predict wait times to manage crowd flow.
* **Model:** **Random Forest Regressor** (Chosen over Linear Regression due to non-linear crowd dynamics).
* **Performance:** Reduced RMSE from ~200 min (raw data) to **9.83 min** (cleaned).
* [cite_start]**Key Insight:** Time of day and specific attractions (e.g., *Wadon*) drive 71% of congestion[cite: 501].

### 2. 🍔 Market Basket Analysis (Sales)
* **Goal:** Design high-conversion menu bundles.
* **Algorithm:** **Apriori Algorithm** (Association Rule Mining).
* **Key Findings:**
    * [cite_start]*The Social Cluster:* Beer, Wine, Nachos, and Pretzels show high lift (3.77)[cite: 674].
    * [cite_start]*The Healthy Cluster:* Salad and Juice have a correlation of 0.81[cite: 615].

### 3. 🎫 Customer Segmentation (Marketing)
* **Goal:** Predict "Gold" vs. "Silver" pass purchases.
* **Model:** **XGBoost Classifier**.
* [cite_start]**Performance:** Achieved **86% Accuracy** in distinguishing pass tiers[cite: 696].
* [cite_start]**Key Driver:** "Club Member" status was the #1 predictor (38% importance)[cite: 745].

## 📂 Project Structure
```bash
├── modules/
│   ├── wait_time_prediction.py   # Random Forest for operations (Maulik)
│   ├── menu_basket_analysis.py   # Apriori algorithm for sales (Mehmet)
│   └── customer_segmentation.py  # XGBoost for customers (Kasun)
├── reports/
│   └── Executive_Presentation.pdf # Final consulting deck
└── requirements.txt
```
🚀 Strategic Recommendations
Based on the data, we recommended the board implement:


Virtual Queues for the "Wadon" ride to disperse the 16% congestion driver.


"Social Combo" Bundles (Nachos + Beer) to target high-lift groups.


Dynamic Staffing shifts to cover the 11:00–15:00 peak window.

👨‍💻 Contributors
Maulik Dilipbhai Chopda - Wait Time Analysis & Operational Strategy

Mehmet Fatih Özdemir - Market Basket Analysis

Kasun Gayashan Pinto Ranwalage - Customer Classification
