# Gourmet Haven Marketing Analytics

## Project Overview
This repository contains a comprehensive marketing analytics solution for **Gourmet Haven**, a gourmet food retailer. The project leverages machine learning and customer segmentation to optimize marketing campaign ROI, identify high-value customer segments, and provide data-driven strategic recommendations.

The analysis includes full-cycle data science work: from data preparation and exploratory analysis to predictive modeling, model interpretation, and a detailed cost-benefit analysis for implementation.

## Business Problem
Gourmet Haven seeks to improve the efficiency and effectiveness of its marketing campaigns. The primary goals are:
- **Increase campaign response rates** through better targeting.
- **Identify key customer segments** with the highest lifetime value.
- **Understand the primary drivers** behind campaign acceptance.
- **Provide a clear, financially-justified roadmap** for marketing technology investments.

## Solution & Key Insights
Our data-driven approach delivered several critical insights:

- **💰 Income is a Key Indicator:** Customers earning over \$75k show **2-3x higher campaign response rates**.
- **🍷 Product Predicts Performance:** **Wine purchases** are the single strongest predictor of campaign acceptance.
- **📊 Recency & Loyalty Matter:** Recent purchasers and long-tenured customers are significantly more likely to respond.
- **🛍️ Channel Preferences Vary:** The **Store** channel has the most customers, but the **Catalog** channel has the highest response rate (36.9%).
- **🧬 Household Composition Influences Spend:** Households with teens have distinct product preferences, particularly for meat and fruits.


## Technical Implementation

### Data & Tools
- **Language:** R
- **Key Libraries:** `tidyverse`, `caret`, `xgboost`, `SHAPforxgboost`, `ROCR`
- **Data:** Customer marketing data (1,792 observations, 26+ features)

### Modeling
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Target Variable:** Campaign Response (Binary Classification)
- **Key Performance Metric:**
  - **AUC (Area Under Curve): 0.89** - Excellent predictive discrimination.
- **Model Interpretability:** SHAP (SHapley Additive exPlanations) analysis was used to explain model predictions and identify feature importance.

### Feature Engineering
Created 11 new features to enhance model performance, including:
- `Total_Spend`: Sum of spending across all product categories.
- `Life_Stage`: Customer segmentation based on marital status.
- `Preferred_Channel`: Primary purchase channel (Web, Catalog, Store).
- `Age_Group`: Binned age categories for analysis.

## Key Outputs
1.  **Customer Segmentation Analysis:** Detailed profiles of high-value customer segments.
2.  **Campaign Response Predictor:** An XGBoost model that scores customers on their likelihood to accept a campaign.
3.  **Strategic Recommendations:** A list of actionable marketing strategies.
4.  **Cost-Benefit Analysis:** A detailed 3-year financial projection for implementing the recommended initiatives, showing a strong ROI for a focused approach.

## How to Run the Analysis

### Prerequisites
- R (version 4.0 or higher)
- RStudio (recommended)

### Installation
1.  Clone this repository:
    ```bash
    git clone https://github.com/[your-username]/Gourmet-Haven-Marketing-Analytics.git
    ```
2.  Open the `GOURMET_HAVEN_MARKETING_ANALYTICS.R` script in RStudio.
3.  Install the required R packages by running the following in your R console:
    ```r
    required_packages <- c("tidyverse", "caret", "xgboost", "Matrix", "ROCR", "scales", "knitr", "SHAPforxgboost")
    install.packages(required_packages)
    ```
4.  The script will guide you to load the source data file (`gourmet_haven_data.csv` or similar) via a file dialog when executed.

## Results & Business Impact
The final recommendation is to implement a phased approach, starting with the **Top 3 Initiatives**:
1.  **Customer Segmentation Platform**
2.  **Personalized Campaign System**
3.  **Multi-Channel Optimization**

This strategy is projected to achieve **~85% of the total benefits** with only **~55% of the investment** required for a full 6-initiative rollout, resulting in an estimated **ROI of over 60%** and a payback period of under two years.


## License
This project is for academic/portfolio purposes. The data is assumed to be proprietary.
