# GOURMET HAVEN MARKETING ANALYTICS

# 0. SETUP: LOAD LIBRARIES & SET OPTIONS
library(tidyverse)    # Data manipulation and visualization
library(caret)        # Machine learning framework
library(xgboost)      # Gradient boosting algorithm
library(Matrix)       # Sparse matrix operations
library(ROCR)         # Model evaluation metrics
library(scales)       # Formatting for plots and tables
library(knitr)        # Professional table formatting
library(SHAPforxgboost) # Model interpretability

# Set global options for reproducibility and display
options(scipen = 999, digits = 4)
set.seed(99) # For reproducible results

# 1. DATA PREPARATION & FEATURE ENGINEERING

load_gourmet_data <- function() {
  # Load and initial data quality check
  cat("Loading Gourmet Haven dataset...\n")
  gourmet_data <- read.csv(choose.files(), header = TRUE)
  
  cat("Initial data dimensions:", dim(gourmet_data), "\n")
  cat("Missing values by column:\n")
  print(colSums(is.na(gourmet_data)))
  
  return(gourmet_data)
}

clean_and_transform <- function(data) {
  # Remove duplicates and handle missing values
  data_clean <- data |> 
    distinct() |> 
    mutate(Income = ifelse(is.na(Income), mean(Income, na.rm = TRUE), Income))
  
  # Standardize column names for clarity
  data_clean <- data_clean |> 
    rename(
      Wines = MntWines,
      Fruits = MntFruits, 
      Meat = MntMeatProducts,
      Fish = MntFishProducts,
      Sweet = MntSweetProducts,
      Gold_Prods = MntGoldProds,
      WebPurchases = NumWebPurchases,
      CatalogPurchases = NumCatalogPurchases,
      StorePurchases = NumStorePurchases,
      WebVisits = NumWebVisitsMonth
    )
  
  # Feature engineering
  data_clean <- data_clean |> 
    mutate(
      Age = 2024 - Year_Birth,
      Total_Spend = Wines + Fruits + Meat + Fish + Sweet + Gold_Prods,
      Total_Purchases = WebPurchases + CatalogPurchases + StorePurchases,
      # Create meaningful customer segments
      Life_Stage = case_when(
        Marital_Status %in% c("Married", "Together") ~ "Couples",
        Marital_Status %in% c("Single", "Divorced", "Widow") ~ "Single",
        Marital_Status %in% c("YOLO", "Absurd", "Alone") ~ "Other",
        TRUE ~ as.character(Marital_Status)
      ),
      Has_Children = as.factor(ifelse(Kidhome > 0 | Teenhome > 0, "Yes", "No")),
      # Recode target variable for clarity
      Campaign_Response = as.factor(ifelse(Response == 1, "Accepted", "Declined")),
      # Create age groups for analysis
      Age_Group = cut(Age,
                      breaks = c(0, 35, 50, 65, 100),
                      labels = c("18-35", "36-50", "51-65", "65+"),
                      include.lowest = TRUE),
      # Calculate channel preferences
      Web_Share = WebPurchases / Total_Purchases,
      Catalog_Share = CatalogPurchases / Total_Purchases,
      Store_Share = StorePurchases / Total_Purchases,
      Preferred_Channel = case_when(
        Web_Share >= Catalog_Share & Web_Share >= Store_Share ~ "Web",
        Catalog_Share >= Web_Share & Catalog_Share >= Store_Share ~ "Catalog",
        TRUE ~ "Store"
      )
    )
  
  # Remove original temporal variables
  data_clean <- data_clean |> 
    select(-c(Year_Birth, Dt_Customer, Response))
  
  return(data_clean)
}

# Execute data preparation pipeline
Gourmet_Haven <- load_gourmet_data() |> clean_and_transform()

cat("Final dataset dimensions:", dim(Gourmet_Haven), "\n")
cat("Data cleaning completed successfully.\n")

# 2. EXPLORATORY DATA ANALYSIS & INSIGHTS

# 2.1 Customer Segmentation by Income
income_analysis <- Gourmet_Haven |>
  mutate(
    Income_Group = cut(Income,
                       breaks = c(0, 25000, 50000, 75000, 100000, Inf),
                       labels = c("0-25k", "25k-50k", "50k-75k", "75k-100k", "100k+"),
                       include.lowest = TRUE)
  ) |>
  group_by(Income_Group) |>
  summarise(
    Customers = n(),
    Avg_Spend = mean(Total_Spend),
    Response_Rate = mean(Campaign_Response == "Accepted"),
    .groups = "drop"
  ) |>
  mutate(Customer_Share = Customers / sum(Customers))

# Display income segmentation results
income_analysis |>
  mutate(
    Avg_Spend = dollar(Avg_Spend, accuracy = 1),
    Response_Rate = percent(Response_Rate, accuracy = 0.1),
    Customer_Share = percent(Customer_Share, accuracy = 0.1)
  ) |>
  kable(caption = "Customer Behavior by Income Segment", 
        align = "lrrrr",
        col.names = c("Income Group", "Customers", "Avg Spend", "Response Rate", "Share"))

# Visualization: Response Rate by Income
ggplot(income_analysis, aes(x = Income_Group, y = Response_Rate, group = 1)) +
  geom_line(color = "darkblue", linetype = "dashed", size = 1.2) +
  geom_point(color = "blue", size = 3) +
  scale_y_continuous(labels = percent) +
  labs(
    title = "Campaign Response Rate Increases with Income",
    subtitle = "Higher income segments show greater campaign engagement",
    x = "Income Group", 
    y = "Response Rate"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "lightgrey", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

# 2.2 Age Group Analysis - Spending Patterns
age_analysis <- Gourmet_Haven |>
  group_by(Age_Group) |>
  summarise(
    Customers = n(),
    Avg_Total_Spend = mean(Total_Spend),
    Avg_Income = mean(Income),
    Response_Rate = mean(Campaign_Response == "Accepted"),
    Avg_Wines = mean(Wines),
    Avg_Meat = mean(Meat),
    Avg_Gold = mean(Gold_Prods),
    .groups = "drop"
  ) |>
  mutate(Customer_Share = Customers / sum(Customers))

# Display age group analysis
age_analysis |>
  mutate(
    Avg_Total_Spend = dollar(Avg_Total_Spend, accuracy = 1),
    Avg_Income = dollar(Avg_Income, accuracy = 1),
    Response_Rate = percent(Response_Rate, accuracy = 0.1),
    Customer_Share = percent(Customer_Share, accuracy = 0.1)
  ) |>
  kable(caption = "Customer Behavior by Age Group", 
        align = "lrrrrrrr",
        col.names = c("Age Group", "Customers", "Avg Spend", "Avg Income", "Response Rate", 
                      "Avg Wines", "Avg Meat", "Avg Gold", "Share"))

# Visualization: Spending by Age Group
age_spend_long <- Gourmet_Haven |>
  select(Age_Group, Wines, Fruits, Meat, Fish, Sweet, Gold_Prods) |>
  pivot_longer(
    cols = -Age_Group,
    names_to = "Product_Category",
    values_to = "Spend"
  ) |>
  group_by(Age_Group, Product_Category) |>
  summarise(Avg_Spend = mean(Spend), .groups = "drop")

ggplot(age_spend_long, aes(x = Age_Group, y = Avg_Spend, fill = Product_Category)) +
  geom_col(position = "dodge", alpha = 0.9, width = 0.8) +
  scale_fill_manual(
    values = c(
      "Wines" = "red",        
      "Fruits" = "darkgreen",     
      "Meat" = "darkred",         
      "Fish" = "blue",         
      "Sweet" = "#FFC0CB",      
      "Gold_Prods" = "#F9C80E"
    )
  ) +
  scale_y_continuous(labels = dollar, expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "Average Spending by Product Category and Age Group",
    subtitle = "Across age demographics, expenditures on meat and wine dominate overall product spending",
    x = "Age Group", 
    y = "Average Spend ($)",
    fill = "Product Category"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.background = element_rect(fill = "grey95", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title = element_text(face = "bold", size = 11),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = "grey40", hjust = 0.5),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.6, "cm"),
    legend.background = element_rect(fill = "grey98", color = "grey80")
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))

# 2.3 Purchase Channel Analysis
channel_analysis <- Gourmet_Haven |>
  summarise(
    Total_Web_Purchases = sum(WebPurchases),
    Total_Catalog_Purchases = sum(CatalogPurchases),
    Total_Store_Purchases = sum(StorePurchases),
    Avg_Web_Purchases = mean(WebPurchases),
    Avg_Catalog_Purchases = mean(CatalogPurchases),
    Avg_Store_Purchases = mean(StorePurchases)
  ) |>
  pivot_longer(
    cols = everything(),
    names_to = "Metric",
    values_to = "Value"
  )

# Channel preference by customer segment
channel_by_segment <- Gourmet_Haven |>
  group_by(Preferred_Channel) |>
  summarise(
    Customers = n(),
    Avg_Income = mean(Income),
    Avg_Total_Spend = mean(Total_Spend),
    Avg_Age = mean(Age),
    Response_Rate = mean(Campaign_Response == "Accepted"),
    .groups = "drop"
  ) |>
  mutate(Channel_Share = Customers / sum(Customers))

# Display channel analysis
channel_by_segment |>
  mutate(
    Avg_Income = dollar(Avg_Income, accuracy = 1),
    Avg_Total_Spend = dollar(Avg_Total_Spend, accuracy = 1),
    Response_Rate = percent(Response_Rate, accuracy = 0.1),
    Channel_Share = percent(Channel_Share, accuracy = 0.1)
  ) |>
  kable(caption = "Customer Profile by Preferred Purchase Channel", 
        align = "lrrrrr",
        col.names = c("Channel", "Customers", "Avg Income", "Avg Spend", "Avg Age", "Response Rate", "Share"))

# Visualization: Channel Preference Distribution
ggplot(channel_by_segment, aes(x = Preferred_Channel, y = Customers, fill = Preferred_Channel)) +
  geom_col(alpha = 0.8) +
  scale_fill_manual(values = c("Web" = "steelblue", "Catalog" = "darkorange", "Store" = "forestgreen")) +
  labs(
    title = "Customer Distribution by Preferred Purchase Channel",
    subtitle = "Store purchases are the most popular channel among customers",
    x = "Preferred Channel", 
    y = "Number of Customers",
    fill = "Channel"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "lightgrey", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

# 2.4 In-Store Product Analysis
# Calculate average spend per product category for in-store purchasers
high_store_customers <- Gourmet_Haven |>
  filter(StorePurchases > 0)  # Only customers who shop in-store

store_product_analysis <- high_store_customers |>
  summarise(
    Avg_Wines_Store = mean(Wines),
    Avg_Fruits_Store = mean(Fruits),
    Avg_Meat_Store = mean(Meat),
    Avg_Fish_Store = mean(Fish),
    Avg_Sweet_Store = mean(Sweet),
    Avg_Gold_Store = mean(Gold_Prods),
    Total_Customers = n()
  ) |>
  pivot_longer(
    cols = starts_with("Avg_"),
    names_to = "Product_Category",
    values_to = "Avg_Spend"
  ) |>
  mutate(
    Product_Category = gsub("Avg_|_Store", "", Product_Category),
    Product_Category = gsub("_", " ", Product_Category)
  )

# Display in-store product analysis
store_product_analysis |>
  mutate(
    Avg_Spend = dollar(Avg_Spend, accuracy = 1),
    Product_Category = str_to_title(Product_Category)
  ) |>
  kable(caption = "Average In-Store Spending by Product Category", 
        align = "lr",
        col.names = c("Total Customers","Product Category", "Average Spend"))

# Visualization: In-Store Product Preferences
ggplot(store_product_analysis, aes(x = reorder(Product_Category, Avg_Spend), y = Avg_Spend)) +
  geom_col(fill = "blue", alpha = 0.8) +
  #coord_flip() +
  scale_y_continuous(labels = dollar) +
  labs(
    title = "Average In-Store Spending by Product Category",
    subtitle = "Wines and Meat products dominate in-store purchases",
    x = "Product Category", 
    y = "Average Spend ($)"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "lightgrey", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

# 2.5 Product Category Analysis by Household Composition
product_categories <- c("Wines", "Fruits", "Meat", "Fish", "Sweet", "Gold_Prods")

product_analysis <- Gourmet_Haven |>
  mutate(Household_Type = ifelse(Teenhome > 0, "With Teens", "Without Teens")) |>
  pivot_longer(
    cols = all_of(product_categories),
    names_to = "Product_Category",
    values_to = "Spend"
  ) |>
  group_by(Product_Category, Household_Type) |>
  summarise(Avg_Spend = mean(Spend), .groups = "drop")

# Visualization: Product Preferences by Household Type
ggplot(product_analysis, aes(x = Product_Category, y = Avg_Spend, fill = Household_Type)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("With Teens" = "blue", "Without Teens" = "darkblue")) +
  scale_y_continuous(labels = dollar) +
  labs(
    title = "Product Category Spending by Household Composition",
    subtitle = "Wines lead spending; meat shows the biggest household gap",
    x = "Product Category", 
    y = "Average Spend ($)",
    fill = "Household Type"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "gray90", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

# 3. MODEL DEVELOPMENT: PREDICTING CAMPAIGN RESPONSE

prepare_model_data <- function(data) {
  # Convert categorical variables to factors
  model_data <- data |>
    mutate(across(c(Education, Marital_Status, Kidhome, Teenhome, 
                    AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, 
                    AcceptedCmp4, AcceptedCmp5, Complain, Has_Children,
                    Life_Stage, Campaign_Response, Age_Group, Preferred_Channel), as.factor))
  
  return(model_data)
}

# Prepare data for modeling
model_data <- prepare_model_data(Gourmet_Haven)

# Split data into training and testing sets
train_index <- createDataPartition(model_data$Campaign_Response, 
                                   p = 0.8, 
                                   list = FALSE)

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

colSums(is.na(train_data))
cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

# 3.1 XGBoost Model Training with Cross-Validation
xgb_model <- train(
  Campaign_Response ~ .,
  data = train_data,
  method = "xgbTree",
  verbose = FALSE,
  tuneGrid = expand.grid(
    nrounds = c(50, 100, 150),
    max_depth = c(3, 5, 7),
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.7
  ),
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    sampling = "up"  # Handle class imbalance
  ),
  metric = "ROC"
)

# Display model performance
cat("Best model parameters:\n")
print(xgb_model$bestTune)

cat("Cross-validation results:\n")
print(xgb_model$results)


# 4. MODEL INTERPRETATION & FEATURE IMPORTANCE

# 4.1 XGBoost model Feature Importance
feature_importance <- varImp(xgb_model)$importance |>
  rownames_to_column("Feature") |>
  arrange(desc(Overall)) |>
  slice_head(n = 15)

ggplot(feature_importance, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "blue", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Top 15 Most Important Predictors",
    subtitle = "Features ranked by XGBoost importance score",
    x = "",
    y = "Importance Score"
  ) +
  theme_minimal()+
  theme(
    panel.background = element_rect(fill = "lightgray", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

# 4.2 SHAP Analysis for Model Interpretability

calculate_shap_direct <- function(model, data) {
  # Prepare data exactly as caret did during training
  features <- data |> 
    select(-Campaign_Response) |>
    mutate(across(where(is.factor), as.integer)) |>
    mutate(across(where(is.character), as.integer)) |>
    mutate(across(where(is.logical), as.integer)) |>
    as.matrix()
  
  # Get model and ensure feature alignment
  raw_model <- model$finalModel
  
  # Manually align features
  expected_features <- raw_model$feature_names
  current_features <- colnames(features)
  
  # Add missing features with zeros
  missing_features <- setdiff(expected_features, current_features)
  if (length(missing_features) > 0) {
    for (f in missing_features) {
      features <- cbind(features, 0)
    }
    colnames(features) <- c(current_features, missing_features)
  }
  
  # Reorder to match model
  features <- features[, expected_features, drop = FALSE]
  
  # Calculate SHAP values using direct prediction
  shap_matrix <- predict(
    raw_model,
    newdata = features,
    predcontrib = TRUE
  )
  
  return(shap_matrix)
}

shap_direct <- calculate_shap_direct(xgb_model, train_data)

# Convert to the format needed for SHAP plotting
shap_scores <- shap_direct[, 1:(ncol(shap_direct)-1)]  # Exclude BIAS column
colnames(shap_scores) <- xgb_model$finalModel$feature_names

# Create manual SHAP summary plot
shap_importance <- colMeans(abs(shap_scores)) |> 
  sort(decreasing = TRUE) |>
  head(15)

importance_df <- data.frame(
  Feature = names(shap_importance),
  Importance = as.numeric(shap_importance)
)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "blue", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "SHAP Feature Importance",
    subtitle = "Mean absolute SHAP value - Impact on model predictions",
    x = "",
    y = "Mean |SHAP| Value"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "lightgrey", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold")
  )

# Display top features
cat("\nTOP FEATURES BY SHAP IMPORTANCE \n")
print(importance_df)

# Create dependence plots for top 2 features
top_2_features <- head(names(shap_importance), 2)

# Prepare feature data for dependence plots
train_features_fixed <- train_data |> 
  select(-Campaign_Response) |>
  mutate(across(where(is.factor), as.integer)) |>
  mutate(across(where(is.character), as.integer)) |>
  mutate(across(where(is.logical), as.integer)) |>
  as.matrix()

for (feature in top_2_features) {
  # Get feature values and SHAP values for this feature
  feature_values <- train_features_fixed[, feature]
  shap_values_feature <- shap_scores[, feature]
  
  plot_data <- data.frame(
    Feature = feature_values,
    SHAP = shap_values_feature
  )
  
  p <- ggplot(plot_data, aes(x = Feature, y = SHAP)) +
    geom_point(alpha = 0.6, color = "blue") +
    geom_smooth(method = "loess", color = "red", se = TRUE) +
    labs(
      title = paste("SHAP Dependence Plot:", feature),
      subtitle = "How feature values affect model predictions",
      x = feature,
      y = "SHAP Value"
    ) +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "lightgrey", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    )
  
  print(p)
}

cat("\nSHAP ANALYSIS COMPLETED SUCCESSFULLY\n")
cat("Top 2 most influential features:\n")
for (i in 1:2) {
  cat(i, ". ", top_2_features[i], " (importance: ", 
      round(importance_df$Importance[i], 4), ")\n", sep = "")
}

# 5. MODEL EVALUATION & PERFORMANCE METRICS 
generate_predictions <- function(model, test_data) {
  predictions <- predict(model, test_data, type = "prob")
  return(predictions$Accepted)  # Probability of accepting campaign
}

# Generate predictions and evaluate
test_predictions <- generate_predictions(xgb_model, test_data)

# Calculate ROC curve and AUC
roc_prediction <- prediction(test_predictions, 
                             test_data$Campaign_Response, 
                             label.ordering = c("Declined", "Accepted"))

roc_performance <- performance(roc_prediction, "tpr", "fpr")
auc_value <- performance(roc_prediction, "auc")@y.values[[1]]

# Plot ROC Curve
plot(roc_performance, 
     colorize = TRUE, 
     main = paste("ROC Curve - XGBoost Model (AUC =", round(auc_value, 3), ")"),
     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

# Display performance metrics
confusion_matrix <- confusionMatrix(
  predict(xgb_model, test_data),
  test_data$Campaign_Response,
  positive = "Accepted"
)

model_performance_table <- data.frame(
  Metric = c(
    "Area Under Curve (AUC)",
    "Overall Accuracy",
    "Sensitivity (Recall)",
    "Specificity",
    "Precision",
    "F1-Score"#,
#    "Kappa"
  ),
  Value = c(
    round(auc_value, 4),
    round(confusion_matrix$overall["Accuracy"], 4),
    round(confusion_matrix$byClass["Sensitivity"], 4),
    round(confusion_matrix$byClass["Specificity"], 4),
    round(confusion_matrix$byClass["Precision"], 4),
    round(confusion_matrix$byClass["F1"], 4)#,
#    round(confusion_matrix$overall["Kappa"], 4)
  ),
  Interpretation = c(
    "Excellent discrimination if >0.9, Good if >0.8",
    "Overall correct classification rate",
    "Ability to correctly identify positive cases",
    "Ability to correctly identify negative cases", 
    "Accuracy when predicting positive class",
    "Balance between precision and recall"#,
#    "Agreement beyond chance (0-1 scale)"
  )
)

cat("MODEL PERFORMANCE SUMMARY\n")
kable(model_performance_table, align = "lrr",
      col.names = c("Performance Metric", "Value", "Interpretation"),
      caption = "XGBoost Model Evaluation Metrics")


# 6. COMPREHENSIVE COST-BENEFIT ANALYSIS

# Calculate current business baseline
current_performance <- Gourmet_Haven |>
  summarise(
    total_customers = n(),
    current_response_rate = mean(Campaign_Response == "Accepted"),
    avg_customer_spend = mean(Total_Spend),
    campaigns_per_year = 6,
    current_annual_responders = total_customers * current_response_rate * campaigns_per_year,
    current_annual_revenue = current_annual_responders * avg_customer_spend * 0.15,
    customer_acquisition_cost = 85, # Industry average for similar businesses
    current_marketing_spend = current_annual_revenue * 0.4 # 40% of revenue spent on marketing
  )

# Implementation costs with realistic vendor pricing
implementation_costs <- data.frame(
  recommendation = c(
    "Customer Segmentation Platform",
    "Personalized Campaign System",
    "Multi-Channel Optimization",
    "Staff Training & Change Management",
    "A/B Testing Infrastructure",
    "Advanced Analytics & Reporting"
  ),
  category = c("Core Platform", "Campaign Management", "Channel Management", "People", "Optimization", "Analytics"),
  setup_cost = c(15000, 12000, 8000, 10000, 6000, 8000),
  annual_license = c(9000, 7200, 4800, 0, 3600, 6000),
  implementation_services = c(10000, 8000, 5000, 6000, 4000, 6000),
  internal_resource_cost = c(8000, 6000, 4000, 12000, 3000, 5000)
) |>
  mutate(
    total_setup_cost = setup_cost + implementation_services + internal_resource_cost,
    year1_total = total_setup_cost + annual_license,
    year2_total = annual_license * 1.05,
    year3_total = annual_license * 1.05^2,
    implementation_months = c(3, 2, 2, 1, 1, 2),
    roi_priority = c(1, 2, 3, 6, 4, 5) # Priority based on expected ROI
  )

# Benefit projections
benefit_projections <- data.frame(
  benefit_source = c(
    "Improved Targeting Efficiency",
    "Personalization Lift", 
    "Channel Optimization",
    "Testing & Optimization",
    "Reduced Acquisition Cost",
    "Increased Customer Lifetime Value"
  ),
  primary_initiative = c(
    "Customer Segmentation Platform",
    "Personalized Campaign System",
    "Multi-Channel Optimization", 
    "A/B Testing Infrastructure",
    "Staff Training & Change Management",
    "Advanced Analytics & Reporting"
  ),
  conservative_lift = c(0.10, 0.08, 0.06, 0.04, -0.03, 0.07),
  moderate_lift = c(0.15, 0.12, 0.09, 0.06, -0.02, 0.10),
  optimistic_lift = c(0.22, 0.18, 0.14, 0.10, -0.01, 0.15),
  compounding_factor = c(0.08, 0.06, 0.05, 0.04, 0.03, 0.07), # Annual growth from previous improvements
  description = c(
    "Better segmentation reducing wasted spend + customer retention improvements",
    "Personalized content increasing engagement + cross-sell opportunities",
    "Optimal channel mix + improved customer journey across touchpoints",
    "Continuous testing driving incremental improvements + faster learning cycles", 
    "Cost savings from reduced broad-reach advertising + improved efficiency",
    "Increased repeat purchases + higher average order value + reduced churn"
  )
) |>
  mutate(
    conservative_annual_value = current_performance$current_annual_revenue * conservative_lift,
    moderate_annual_value = current_performance$current_annual_revenue * moderate_lift,
    optimistic_annual_value = current_performance$current_annual_revenue * optimistic_lift
  )

# 2. FINANCIAL MODEL WITH COMPOUNDING EFFECTS 

calculate_compounding_financials <- function(initiative_list, scenario_name) {
  
  selected_initiatives <- implementation_costs |>
    filter(recommendation %in% initiative_list)
  
  cost_summary <- selected_initiatives |>
    summarise(
      total_setup_cost = sum(total_setup_cost),
      total_annual_license = sum(annual_license),
      max_implementation_months = max(implementation_months)
    )
  
  # Get benefit sources for selected initiatives
  benefit_sources <- benefit_projections |>
    filter(primary_initiative %in% initiative_list)
  
  # Calculate base annual benefits
  base_conservative <- sum(benefit_sources$conservative_annual_value)
  base_moderate <- sum(benefit_sources$moderate_annual_value)
  base_optimistic <- sum(benefit_sources$optimistic_annual_value)
  
  # Calculate weighted compounding factor based on selected initiatives
  weighted_compounding <- weighted.mean(
    benefit_sources$compounding_factor,
    benefit_sources$moderate_annual_value
  )
  
  # Create financial projection with compounding benefits
  financial_projection <- data.frame(
    year = 0:3,
    period = c("Implementation", "Year 1", "Year 2", "Year 3"),
    # Costs
    implementation_cost = c(cost_summary$total_setup_cost, 0, 0, 0),
    annual_license_cost = c(0, cost_summary$total_annual_license, 
                            cost_summary$total_annual_license * 1.05,
                            cost_summary$total_annual_license * 1.05^2),
    internal_cost = c(length(initiative_list) * 2000, 
                      length(initiative_list) * 1500 * 1.05,
                      length(initiative_list) * 1500 * 1.05^2,
                      length(initiative_list) * 1500 * 1.05^3)
  ) |>
    mutate(total_cost = implementation_cost + annual_license_cost + internal_cost)
  
  # Calculate benefits with compounding effects
  # Year 1: Initial implementation (60% of full potential due to ramp-up)
  year1_conservative <- base_conservative * 0.6
  year1_moderate <- base_moderate * 0.6
  year1_optimistic <- base_optimistic * 0.6
  
  # Year 2: Full implementation + compounding from Year 1 improvements
  year2_conservative <- base_conservative * 0.9 + (year1_conservative * weighted_compounding)
  year2_moderate <- base_moderate * 0.9 + (year1_moderate * weighted_compounding)
  year2_optimistic <- base_optimistic * 0.9 + (year1_optimistic * weighted_compounding)
  
  # Year 3: Full benefits + cumulative compounding from previous years
  year3_conservative <- base_conservative + (year2_conservative * weighted_compounding)
  year3_moderate <- base_moderate + (year2_moderate * weighted_compounding)
  year3_optimistic <- base_optimistic + (year2_optimistic * weighted_compounding)
  
  # Assign benefits to projection
  financial_projection$conservative_benefit <- c(0, year1_conservative, year2_conservative, year3_conservative)
  financial_projection$moderate_benefit <- c(0, year1_moderate, year2_moderate, year3_moderate)
  financial_projection$optimistic_benefit <- c(0, year1_optimistic, year2_optimistic, year3_optimistic)
  
  # Calculate net cash flows and cumulative position
  financial_projection <- financial_projection |>
    mutate(
      net_conservative = conservative_benefit - total_cost,
      net_moderate = moderate_benefit - total_cost,
      net_optimistic = optimistic_benefit - total_cost,
      cumulative_conservative = cumsum(net_conservative),
      cumulative_moderate = cumsum(net_moderate),
      cumulative_optimistic = cumsum(net_optimistic)
    )
  
  # Financial metrics
  metrics <- financial_projection |>
    summarise(
      scenario_name = scenario_name,
      num_initiatives = length(initiative_list),
      total_investment_3yr = sum(total_cost),
      total_benefit_conservative = sum(conservative_benefit),
      total_benefit_moderate = sum(moderate_benefit),
      total_benefit_optimistic = sum(optimistic_benefit),
      weighted_compounding_factor = weighted_compounding,
      
      roi_conservative = (total_benefit_conservative - total_investment_3yr) / total_investment_3yr,
      roi_moderate = (total_benefit_moderate - total_investment_3yr) / total_investment_3yr,
      roi_optimistic = (total_benefit_optimistic - total_investment_3yr) / total_investment_3yr,
      
      payback_conservative = if(any(financial_projection$cumulative_conservative >= 0)) 
        min(which(financial_projection$cumulative_conservative >= 0)) - 1 else NA,
      payback_moderate = if(any(financial_projection$cumulative_moderate >= 0)) 
        min(which(financial_projection$cumulative_moderate >= 0)) - 1 else NA,
      payback_optimistic = if(any(financial_projection$cumulative_optimistic >= 0)) 
        min(which(financial_projection$cumulative_optimistic >= 0)) - 1 else NA,
      
      npv_conservative = sum(financial_projection$net_conservative / (1.1)^financial_projection$year),
      npv_moderate = sum(financial_projection$net_moderate / (1.1)^financial_projection$year),
      npv_optimistic = sum(financial_projection$net_optimistic / (1.1)^financial_projection$year),
      
      benefit_cost_ratio = total_benefit_moderate / total_investment_3yr,
      annualized_roi = (1 + roi_moderate)^(1/3) - 1
    )
  
  return(list(
    metrics = metrics,
    cost_breakdown = selected_initiatives,
    projection = financial_projection,
    benefit_sources = benefit_sources,
    compounding_factor = weighted_compounding
  ))
}

# 3. SCENARIO ANALYSIS

all_initiatives <- implementation_costs$recommendation
top_3_initiatives <- c("Customer Segmentation Platform", "Personalized Campaign System", "Multi-Channel Optimization")

# Calculate both scenarios with compounding effect
all_6_compounding <- calculate_compounding_financials(all_initiatives, "All 6 Initiatives")
top_3_compounding <- calculate_compounding_financials(top_3_initiatives, "Top 3 Initiatives")

# 4. COMPREHENSIVE VISUALIZATIONS

# Visualization 1: Compounding Benefit Growth
benefit_growth_data <- bind_rows(
  all_6_compounding$projection |>
    select(period, benefit = moderate_benefit) |>
    mutate(Scenario = "All 6 Initiatives", Type = "With Compounding"),
  top_3_compounding$projection |>
    select(period, benefit = moderate_benefit) |>
    mutate(Scenario = "Top 3 Initiatives", Type = "With Compounding")
) |>
  mutate(period = factor(period, levels = c("Implementation", "Year 1", "Year 2", "Year 3")))

compounding_growth_plot <- ggplot(benefit_growth_data, aes(x = period, y = benefit, color = Scenario, group = Scenario)) +
  geom_line(size = 2, alpha = 0.8) +
  geom_point(size = 4) +
  geom_text(aes(label = dollar(benefit, accuracy = 1)), vjust = -1, size = 3.5, fontface = "bold") +
  scale_color_manual(values = c("blue", "darkblue")) +
  scale_y_continuous(labels = dollar_format(scale = 0.001, suffix = "K")) +
  labs(
    title = "Compounding Benefit Growth Over Time",
    subtitle = paste("Year-over-year growth from", percent(mean(c(all_6_compounding$compounding_factor, top_3_compounding$compounding_factor)), accuracy = 0.1), "compounding effects"),
    x = "Implementation Period",
    y = "Annual Benefits ($ Thousands)",
    color = "Scenario"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "gray40")
  )

# Visualization 2: ROI Comparison with Compounding
roi_data <- bind_rows(all_6_compounding$metrics, top_3_compounding$metrics)

roi_comparison_plot <- ggplot(roi_data, aes(x = scenario_name, y = roi_moderate, fill = scenario_name)) +
  geom_col(alpha = 0.8, width = 0.6) +
  geom_text(aes(label = percent(roi_moderate, accuracy = 0.1)), vjust = -0.5, size = 6, fontface = "bold") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("blue", "darkblue")) +
  scale_y_continuous(labels = percent, limits = c(0, 1.0)) +
  labs(
    title = "3-Year ROI Comparison with Compounding Effects",
    subtitle = "Includes compounding growth from previous years' improvements",
    x = NULL,
    y = "Return on Investment (ROI)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 12),
    plot.title = element_text(face = "bold", size = 16)
  )

# Visualization 3: Cumulative Cash Flow with Break-even
cumulative_data <- bind_rows(
  all_6_compounding$projection |>
    select(period, cumulative = cumulative_moderate) |>
    mutate(Scenario = "All 6 Initiatives"),
  top_3_compounding$projection |>
    select(period, cumulative = cumulative_moderate) |>
    mutate(Scenario = "Top 3 Initiatives")
) |>
  mutate(period = factor(period, levels = c("Implementation", "Year 1", "Year 2", "Year 3")))

break_even_plot <- ggplot(cumulative_data, aes(x = period, y = cumulative, color = Scenario, group = Scenario)) +
  geom_line(size = 2, alpha = 0.8) +
  geom_point(size = 4) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 1) +
  geom_text(aes(label = dollar(cumulative, accuracy = 1)), vjust = -1, size = 3.5, fontface = "bold") +
  scale_color_manual(values = c("blue", "darkblue")) +
  scale_y_continuous(labels = dollar_format(scale = 0.001, suffix = "K")) +
  labs(
    title = "Cumulative Cash Flow & Break-even Analysis",
    subtitle = "Point where cumulative benefits exceed cumulative costs",
    x = "Implementation Period",
    y = "Cumulative Cash Flow ($ Thousands)",
    color = "Scenario"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Visualization 4: Cost vs Benefit Comparison
cost_benefit_data <- bind_rows(
  all_6_compounding$metrics |>
    select(Scenario = scenario_name, Investment = total_investment_3yr, Benefits = total_benefit_moderate) |>
    mutate(Net = Benefits - Investment),
  top_3_compounding$metrics |>
    select(Scenario = scenario_name, Investment = total_investment_3yr, Benefits = total_benefit_moderate) |>
    mutate(Net = Benefits - Investment)
) |>
  pivot_longer(cols = c(Investment, Benefits, Net), names_to = "Metric", values_to = "Value")

cost_benefit_plot <- ggplot(cost_benefit_data, aes(x = Scenario, y = Value, fill = Metric)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("blue", "darkblue", "steelblue")) +
  scale_y_continuous(labels = dollar_format(scale = 0.001, suffix = "K")) +
  labs(
    title = "3-Year Investment vs Benefits Comparison",
    subtitle = "Total financial impact over implementation period",
    x = NULL,
    y = "Amount ($ Thousands)",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# 5. TABLES 

# Table 1: Executive Summary
executive_summary_table <- bind_rows(all_6_compounding$metrics, top_3_compounding$metrics) |>
  transmute(
    Scenario = scenario_name,
    `Initiatives` = num_initiatives,
    `Compounding Factor` = percent(weighted_compounding_factor, accuracy = 0.1),
    `3-Yr Investment` = dollar(total_investment_3yr),
    `3-Yr Benefits` = dollar(total_benefit_moderate),
    `Net Return` = dollar(total_benefit_moderate - total_investment_3yr),
    `ROI` = percent(roi_moderate, accuracy = 0.1),
    `Payback` = ifelse(is.na(payback_moderate), ">3 yrs", paste(payback_moderate, "yrs")),
    `NPV @10%` = dollar(npv_moderate),
    `Benefit/Cost` = round(benefit_cost_ratio, 2)
  ) |>
  kable(format = "html", caption = "Executive Summary") |>
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"), 
    full_width = FALSE,
    font_size = 14
  ) |>
  row_spec(0, bold = TRUE, color = "white", background = "grey") |>
  column_spec(1, bold = TRUE)

# Table 2: Detailed Cost Breakdown
cost_breakdown_table <- implementation_costs |>
  select(
    Initiative = recommendation,
    Category = category,
    `Software` = setup_cost,
    `Services` = implementation_services,
    `Internal` = internal_resource_cost,
    `Annual License` = annual_license,
    `Total Y1` = year1_total,
    `ROI Priority` = roi_priority
  ) |>
  mutate(across(where(is.numeric), ~ifelse(.x >= 1000, dollar(.x), .x))) |>
  kable(format = "html", caption = "Detailed Implementation Cost Breakdown") |>
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE) |>
  row_spec(0, bold = TRUE, background = "lightgrey")

# Table 3: Benefit Projections with Compounding
benefit_table <- benefit_projections |>
  select(
    `Benefit Source` = benefit_source,
    `Primary Initiative` = primary_initiative,
    `Cons Lift` = conservative_lift,
    `Mod Lift` = moderate_lift,
    `Opt Lift` = optimistic_lift,
    `Compounding` = compounding_factor,
    `Annual Value` = moderate_annual_value
  ) |>
  mutate(
    across(c(`Cons Lift`, `Mod Lift`, `Opt Lift`, `Compounding`), ~percent(., accuracy = 0.1)),
    `Annual Value` = dollar(`Annual Value`)
  ) |>
  kable(format = "html", caption = "Benefit Projections with Compounding Factors") |>
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE) |>
  row_spec(0, bold = TRUE, background = "lightgrey")

# Table 4: Annual Financial Projection (Top 3 Scenario)
annual_projection_table <- top_3_compounding$projection |>
  select(
    Period = period,
    `Implementation Cost` = implementation_cost,
    `License Cost` = annual_license_cost,
    `Internal Cost` = internal_cost,
    `Total Cost` = total_cost,
    `Benefits` = moderate_benefit,
    `Net Cash Flow` = net_moderate,
    `Cumulative` = cumulative_moderate
  ) |>
  mutate(across(where(is.numeric), dollar)) |>
  kable(format = "html", caption = "Annual Financial Projection - Top 3 Initiatives") |>
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE) |>
  row_spec(0, bold = TRUE, background = "lightgrey")

# Table 5: Risk Assessment & Mitigation
risk_table <- data.frame(
  `Risk Factor` = c(
    "Implementation Delays", 
    "Benefit Realization Lag",
    "Staff Adoption Resistance",
    "Technology Integration Issues",
    "Data Quality Problems",
    "Vendor Performance"
  ),
  `Probability` = c("Medium", "Low", "Medium", "Low", "Medium", "Low"),
  `Impact` = c("High", "High", "Medium", "High", "Medium", "Medium"),
  `Mitigation Strategy` = c(
    "Phased implementation with clear milestones",
    "Conservative benefit ramp-up assumptions",
    "Comprehensive training & change management",
    "Technical proof-of-concept before full rollout",
    "Data audit and cleanup during implementation",
    "Multiple vendor options with performance clauses"
  ),
  `Top 3 Advantage` = c("Easier coordination", "Faster learning", "Focused training", 
                        "Simpler integration", "Targeted cleanup", "Fewer vendors")
) |>
  kable(format = "html", caption = "Risk Assessment & Mitigation Strategies") |>
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE) |>
  row_spec(0, bold = TRUE, background = "lightgrey")

# 6. DISPLAY TABLES

executive_summary_table
cost_breakdown_table
benefit_table
annual_projection_table
risk_table

# Display all visualizations
print(compounding_growth_plot)
print(roi_comparison_plot)
print(break_even_plot)
print(cost_benefit_plot)

# Final Recommendation with Compounding Justification
final_recommendation <- data.frame(
  `Recommendation` = "IMPLEMENT TOP 3 INITIATIVES FIRST",
  `Primary Justification` = "Superior financial returns with compounding benefits",
  `Key Metrics` = paste(
    "ROI:", percent(top_3_compounding$metrics$roi_moderate, accuracy = 0.1),
    "| Payback:", ifelse(is.na(top_3_compounding$metrics$payback_moderate), ">3 yrs", 
                         paste(top_3_compounding$metrics$payback_moderate, "yrs")),
    "| NPV:", dollar(top_3_compounding$metrics$npv_moderate)
  ),
  `Compounding Advantage` = paste(
    "Year-over-year growth of", 
    percent(top_3_compounding$compounding_factor, accuracy = 0.1),
    "from sustained improvements"
  ),
  `Implementation Timeline` = "Months 1-6 for core platforms, Months 7-12 for optimization",
  `Year 1 Investment` = dollar(top_3_compounding$cost_breakdown$year1_total |> sum())
) |>
  t() |>
  as.data.frame() |>
  setNames("Value") |>
  kable(format = "html", caption = "FINAL RECOMMENDATION & IMPLEMENTATION ROADMAP") |>
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE) |>
  row_spec(0, bold = TRUE, color = "white", background = "grey")
final_recommendation

