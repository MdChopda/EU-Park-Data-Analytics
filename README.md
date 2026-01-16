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
* **Key Insight:** Time of day and specific attractions (e.g., *Wadon*) drive 71% of congestion.

### 2. 🍔 Market Basket Analysis (Sales)
* **Goal:** Design high-conversion menu bundles.
* **Algorithm:** **Apriori Algorithm** (Association Rule Mining).
* **Key Findings:**
    * *The Social Cluster:* Beer, Wine, Nachos, and Pretzels show high lift (3.77).
    * *The Healthy Cluster:* Salad and Juice have a correlation of 0.81.

### 3. 🎫 Customer Segmentation (Marketing)
* **Goal:** Predict "Gold" vs. "Silver" pass purchases.
* **Model:** **XGBoost Classifier**.
* **Performance:** Achieved **86% Accuracy** in distinguishing pass tiers.
* **Key Driver:** "Club Member" status was the #1 predictor (38% importance).

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
## 🚀 Strategic Recommendations
Based on our data analysis, we propose three high-impact interventions to optimize park performance:

1.  **Implement Virtual Queues for "Wadon":** The "Wadon" attraction is the single largest driver of congestion, accounting for **16% of wait time variance**. A virtual queue will disperse crowds from this bottleneck.
2.  **Launch "Social Combo" Bundles:** Our Market Basket Analysis identified a strong correlation (Lift: 3.77) between **Nachos, Pretzels, Beer, and Wine**. Bundling these items targets the high-value "Social Cluster" to increase revenue.
3.  **Deploy Dynamic Staffing:** The predictive model identified **11:00 – 15:00** as the critical peak window. Shifting staff breaks away from this period will maximize throughput capacity when demand is highest.

## 👨‍💻 Contributors
This project was executed by a specialized data science consulting team:

| Consultant | Role & Focus Area | Key Contribution |
| :--- | :--- | :--- |
| **Maulik Dilipbhai Chopda** | **Wait Time Analysis & Operational Strategy** | Developed the Random Forest model (RMSE 9.83 min) to predict attraction wait times and optimize park logistics. |
| **Mehmet Fatih Özdemir** | **Market Basket Analysis** | Utilized the Apriori algorithm to uncover menu item correlations and design high-conversion food bundles. |
| **Kasun Gayashan Pinto Ranwalage** | **Customer Classification** | Built an XGBoost classifier (86% Accuracy) to segment customers and predict season pass purchasing behavior. |
