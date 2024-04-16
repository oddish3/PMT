#----
# A Poor Means Test Replication Study
#Classification using SVM and XGBoost
#Author : Sol Yates

## TO DO - DOUBLE CHECK confusion matrices, implement SVR + CART?

rm(list=ls())
# setwd("/home/oddish3/Documents/R_folder/UG/PMT/Scripts")

# Required libraries
library(haven)
library(e1071)
library(magrittr)
library(dplyr)
library(caTools)
library(ggplot2)
library(xgboost)
library(caret)
library(pROC)
library(dplyr)
#library(smotefamily)
library(xtable)
library(rBayesianOptimization)

data <- readRDS("data/out/data_over_sampled.rds")
# data <- data %>% select(c("poor_20", "toilet_pit", "wall_finish",
#                            "fuel_elecgas", "fuel_charcoal","urban", "female_head",
#                            "edu_head_primary", "edu_head_secondary", "div_sep_head", "widow_head", 
#                            "work_paid_head", "work_selfemp_nonf_head"))#, "muslim", "christian"))
# 

set.seed(122)
# Train/test split 80% train, 20% test
data$poor_20 <- factor(data$poor_20, levels = c("0", "1"), labels = c("Class0", "Class1"))
train_index <- createDataPartition(data$poor_20, p = 0.8, list = FALSE, times = 1)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#### SVM --------
# Basic SVM ----

# kernels to iterate through
kernels <- c("linear", "radial", "polynomial", "sigmoid")

# Initialise
metrics_table <- data.frame(Kernel = character(), Accuracy = numeric(), Precision = numeric(),
                            sensitivity = numeric(), F1_Score = numeric(), AUC = numeric(),
                            stringsAsFactors = FALSE)

# Iterate through each kernel
for (kernel in kernels) {
  formula <- poor_20 ~ toilet_pit + wall_finish +
    fuel_elecgas + fuel_charcoal + urban + female_head +
    edu_head_primary + edu_head_secondary + div_sep_head + widow_head +
    work_paid_head + work_selfemp_nonf_head

  # Train the SVM model
  svm_model <- svm(formula, data = train_data, kernel = kernel, cost = 1, probability = TRUE)
  
  # Predict on test data
  preds_prob <- predict(svm_model, test_data, type = "probabilities")

  # Confusion Matrix
  cm <- table(test_data$poor_20, preds_prob, dnn = c("Actual", "Predicted"))

  # Calculate metrics
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- cm[2, 2] / sum(cm[, 2])
  sensitivity <- cm[2, 2] / sum(cm[2, ])
  f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))
  roc_obj <- roc(test_data$poor_20, as.numeric(preds_prob))
  auc <- auc(roc_obj)

  # Add metrics to the table
  metrics_table <- rbind(metrics_table, data.frame(Kernel = kernel, Accuracy = accuracy,
                                                   Precision = precision, sensitivity = sensitivity,
                                                   F1_Score = f1_score, AUC = auc))
}

# Print the metrics table
print(metrics_table)
xtable(cm)
xtable(metrics_table)

#### radial kernel w some tuning  ----

# Define a smaller hyperparameter grid
hyperparameters <- expand.grid(
  C = seq(0.1, 1, by = 0.1),
  sigma = c(0.01, 0.1)  # Smaller sigma values
)

# Specify the features formula
features_formula <- formula(paste("poor_20 ~ toilet_pit + wall_finish + fuel_elecgas + fuel_charcoal + urban + female_head + edu_head_primary + edu_head_secondary + div_sep_head + widow_head + work_paid_head + work_selfemp_nonf_head"))

# Perform grid search with cross-validation
ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
svm_grid <- train(features_formula, data = train_data, method = "svmRadial", tuneGrid = hyperparameters, trControl = ctrl, preProcess = c("center", "scale"))

# Get the best SVM model
best_svm_model <- svm_grid$finalModel
test_preds <- predict(svm_model, newdata = test_data, probability = TRUE)
test_preds_prob <- attr(test_preds, "probabilities")[, "Class1"]
cm <- table(test_data$poor_20, ifelse(test_preds_prob > 0.5, 1, 0), dnn = c("Actual", "Predicted"))
accuracy <- sum(diag(cm)) / sum(cm)
precision <- cm[2, 2] / sum(cm[, 2])
sensitivity <- cm[2, 2] / sum(cm[2, ])
f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))

# Calculate ROC AUC
roc_obj <- roc(test_data$poor_20, test_preds_prob)
auc <- auc(roc_obj)

# Create a data frame to store the metrics
metrics_table <- data.frame(Kernel = "Radial", Cost = svm_model$cost, Gamma = svm_model$gamma,
                            Accuracy = accuracy, Precision = precision, Sensitivity = sensitivity, F1_Score = f1_score, AUC = auc)

# Print the metrics table
print(metrics_table)
cm
xtable(cm)
xtable(metrics_table)


########## not working. bayesian optimmisation script ------
# svm_fun <- function(cost, gamma) {
#   tryCatch({
#     model <- svm(poor_20 ~ ., data = train_data, kernel = "radial",
#                  cost = cost, gamma = gamma, probability = TRUE)
#     
#     val_predictions <- predict(model, val_data, type = "response")
#     cm <- confusionMatrix(val_predictions, val_data$poor_20)
#     precision_val <- cm$byClass['Positive Predictive Value']
#     
#     # Return a very low score if precision is NA
#     if (is.na(precision_val)) return(list(Score = -0.5, Pred = val_predictions))
#     return(list(Score = precision_val, Pred = val_predictions))
#   }, error = function(e) {
#     # Return a very low score in case of error
#     return(list(Score = -1, Pred = NULL))
#   })
# }
# 
# # Narrowing down the hyperparameter ranges based on typical values for SVM
# bounds <- list(cost = c(0.1, 5), gamma = c(0.01, 0.5))
# 
# # Run Bayesian Optimization again
# svm_opt <- BayesianOptimization(svm_fun, 
#                                 bounds = bounds, 
#                                 init_points = 5, 
#                                 n_iter = 20)
# # Train the final SVM model with the best parameters
# best_svm <- svm(poor_20 ~ ., data = train_data, kernel = "radial",
#                 cost = svm_opt$Best_Par$cost, 
#                 gamma = svm_opt$Best_Par$gamma, 
#                 probability = TRUE)
# 
# test_predictions <- predict(best_svm, test_data, type = "response")
# 
# # Confusion Matrix
# cm <- confusionMatrix(test_predictions, test_data$poor_20)
# 
# # Calculating metrics
# recall <- cm$byClass['Sensitivity']
# precision <- cm$byClass['Positive Predictive Value']
# f1_score <- 2 * ((precision * recall) / (precision + recall))
# 
# # Output the metrics
# print(paste("Recall:", recall))
# print(paste("Precision:", precision))
# print(paste("F1 Score:", f1_score))
# 
# # PR-AUC
# prob_predictions <- predict(best_svm, test_data, type = "probabilities")[,2]
# pr <- pr.curve(scores.class0 = prob_predictions, weights.class0 = test_data$poor_20 == "1",
#                curve = TRUE)
# plot(pr)



##### exploring sigmoid kernel ----

# Define a custom model

svmSigmoidModel <- list(
  type = "Classification",
  library = "e1071",
  loop = NULL,
  method = "svmSigmoid",
  parameters = data.frame(parameter = c('cost', 'gamma'), class = c('numeric', 'numeric')),
  grid = function(x, y, len = NULL, search = "grid") {
    if(search == "grid") {
      expand.grid(cost = 10^(-1:2), gamma = 10^(-2:1))
    } else {
      data.frame(cost = runif(30, -2, 2), gamma = runif(30, -2, 2))
    }
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    svm(x = x, y = y, kernel = "sigmoid", cost = param$cost, gamma = param$gamma, probability = TRUE, ...)
  },
  predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    predict(modelFit, newdata)
  },
  prob = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    preds <- predict(modelFit, newdata, probability = TRUE)
    attr(preds, "probabilities")
  },
  predictors = function(x, ...) {
    model.matrix(~ . -1, data = as.data.frame(x))
  },
  varImp = NULL,
  levels = function(x) levels(x$y),
  tags = c("SVM", "Kernel", "Sigmoid")
)


# Define the control using a cross-validation approach
train_control <- trainControl(method = "cv", number = 10,
                              summaryFunction = twoClassSummary,  # Function to use AUC
                              classProbs = TRUE,  # Probabilities required for AUC
                              selectionFunction = "best")

# Define the grid for tuning
grid <- expand.grid(cost = 10^(-1:2), gamma = 10^(-2:1))

# Train the model with AUC as the metric ########### 

## --------------------------

#####  Warning - Following takes a short while to run  

## --------------------

svm_model <- train(poor_20 ~ ., data = train_data, method = svmSigmoidModel,
                   trControl = train_control, tuneGrid = grid, preProcess = c("center", "scale"),
                   metric = "ROC")  # Specify ROC as the metric for model tuning

preds_prob <- predict(svm_model, test_data, type = "prob")

threshold <- 0.5  # can change this threshold as needed
predicted_classes <- ifelse(preds_prob[, "Class1"] > threshold, "Class1", "Class0")

# Confusion Matrix
cm <- table(test_data$poor_20, predicted_classes, dnn = c("Actual", "Predicted"))

# Calculate metrics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- cm[2, 2] / sum(cm[, 2])
sensitivity <- cm[2, 2] / sum(cm[2, ])
f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))
roc_obj <- roc(test_data$poor_20, preds_prob$Class1)
auc <- auc(roc_obj)

# Add metrics to the table
metrics_table <- rbind(data.frame(Accuracy = accuracy,
                                                 Precision = precision, sensitivity = sensitivity,
                                                 F1_Score = f1_score, AUC = as.numeric(auc)))

# Print the metrics table -- slight improvement in AUC
print(metrics_table)
print(cm)
xtable(cm)
xtable(metrics_table)



### XGBoost ----
# Basic XGBoost Model Training and Prediction

# Convert character variables to numeric
data <- mutate_if(data, is.character, as.numeric)
data$poor_20 <- ifelse(data$poor_20 == "Class1", 1, 0)
set.seed(1234)

# Stratified sampling based on 'poor_20'
trainIndex <- createDataPartition(data$poor_20, p = 0.8, list = FALSE, times = 1)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Prepare data in DMatrix format
trainData_matrix <- xgb.DMatrix(data = as.matrix(trainData[-which(names(trainData) == "poor_20")]),
                                label = trainData$poor_20)
testData_matrix <- xgb.DMatrix(data = as.matrix(testData[-which(names(testData) == "poor_20")]),
                               label = testData$poor_20)

# Parameters for XGBoost
params <- list(
  objective = "binary:logistic",
  eta = 1,
  max_depth = 2,  
  nthread = 2
)

# Train the XGBoost model
bstSparse <- xgboost(params = params, data = trainData_matrix, nrounds = 100)

# Predict on test data (probabilities)
preds_prob <- predict(bstSparse, testData_matrix)

# Convert probabilities to binary class (0 or 1), assuming 0.5 as threshold
preds_binary <- ifelse(preds_prob > 0.5, 1, 0)

# Confusion Matrix
cm <- table(testData$poor_20, preds_binary)

# Calculate metrics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- cm[2, 2] / sum(cm[, 2])
sensitivity <- cm[2, 2] / sum(cm[2, ])
f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))
roc_obj <- roc(testData$poor_20, preds_prob)
auc <- auc(roc_obj)

# Create a data frame to store the metrics
metrics_table <- data.frame(Accuracy = accuracy, Precision = precision, 
                            sensitivity = sensitivity, F1_Score = f1_score, AUC = auc)

# Print the metrics table
print(metrics_table)
cm
xtable(metrics_table)
xtable(cm)

### look at optimal threshold (was 0.5) -----------------

# Finding optimal threshold for maximizing sensitivity
roc_obj <- roc(testData$poor_20, preds_prob)
optimal_threshold <- coords(roc_obj, "best", ret="threshold")
preds_binary_optimal <- ifelse(preds_prob > as.numeric(optimal_threshold), 1, 0)

# Confusion Matrix with Optimal Threshold
cm_optimal <- table(testData$poor_20, preds_binary_optimal)

# Calculate metrics with Optimal Threshold
accuracy_optimal <- sum(diag(cm_optimal)) / sum(cm_optimal)
precision_optimal <- cm_optimal[2, 2] / sum(cm_optimal[, 2])
sensitivity_optimal <- cm_optimal[2, 2] / sum(cm_optimal[2, ])
f1_score_optimal <- 2 * ((precision_optimal * sensitivity_optimal) / (precision_optimal + sensitivity_optimal))
auc_optimal <- auc(roc_obj)

# Create a data frame to store the metrics
metrics_table_optimal <- data.frame(Threshold = optimal_threshold, 
                                    Accuracy = accuracy_optimal, 
                                    Precision = precision_optimal, 
                                    Sensitivity = sensitivity_optimal, 
                                    F1_Score = f1_score_optimal, 
                                    AUC = auc_optimal)

# Print the metrics table with Optimal Threshold
print(metrics_table_optimal)
xtable(metrics_table_optimal)
print(cm_optimal)
xtable(cm_optimal)

### grid search for highest auc ----        ########### Warning, also takes short while tho slightly less
# Define the parameter grid
grid <- expand.grid(
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1, 2),
  nrounds = c(50, 100, 150)
)

# Function to perform grid search
grid_search <- function(params, dtrain, folds) {
  best_auc <- 0
  best_params <- NULL
  for(i in 1:nrow(params)) {
    # Extract parameters
    param <- list(
      objective = "binary:logistic",
      max_depth = params$max_depth[i],
      eta = params$eta[i],
      gamma = params$gamma[i]
    )
    
    # Cross-validation
    cv <- xgb.cv(param, dtrain, nrounds = params$nrounds[i], nfold = folds, 
                 metrics = "auc", showsd = TRUE, stratified = TRUE, print_every_n = 10)
    
    # Update best score
    mean_auc <- max(cv$evaluation_log$test_auc_mean)
    if(mean_auc > best_auc) {
      best_auc <- mean_auc
      best_params <- param
      best_params$nrounds <- params$nrounds[i]
    }
  }
  list(best_params = best_params, best_auc = best_auc)
}

# Perform grid search
folds <- 5
best_model <- grid_search(grid, trainData_matrix, folds)

# Print best parameters
print(best_model$best_params)
print(paste("Best AUC:", best_model$best_auc))

# Final Model Training with Best Parameters
final_params <- best_model$best_params

final_model <- xgboost(params = final_params, data = trainData_matrix, nrounds = 100, verbose = 0)

# Predict on test data
final_preds <- predict(final_model, testData_matrix)

# ROC and AUC for final model
roc_obj_final <- roc(testData$poor_20, final_preds)
auc_value_final <- auc(roc_obj_final)
print(paste("AUC for Final Model:", auc_value_final))

# Convert predictions to binary class (0 or 1), assuming 0.5 as threshold
final_preds_binary <- ifelse(final_preds > 0.5, 1, 0)

# Confusion Matrix for final model
cm_final <- table(testData$poor_20, final_preds_binary)

# Evaluation Metrics Calculation
accuracy_final <- sum(diag(cm_final)) / sum(cm_final)
precision_final <- cm_final[2, 2] / sum(cm_final[, 2])
sensitivity_final <- cm_final[2, 2] / sum(cm_final[2, ])
f1_score_final <- 2 * ((precision_final * sensitivity_final) / (precision_final + sensitivity_final))

# Create a data frame to store the metrics
metrics_table_final <- data.frame(Accuracy = accuracy_final, 
                                  Precision = precision_final, 
                                  sensitivity = sensitivity_final, 
                                  F1_Score = f1_score_final, 
                                  AUC = auc_value_final)

# Print the metrics table with Optimal Threshold
print(metrics_table_final)
xtable(metrics_table_final)
print(cm_final)
xtable(cm_final)

### grid search for highest sensitivity ----

# Define the parameter grid
grid <- expand.grid(
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1, 2),
  nrounds = c(50, 100, 150)
)

# Function to perform grid search focusing on sensitivity
grid_search_sensitivity <- function(params, dtrain, dtest, actual, folds) {
  best_sensitivity <- 0
  best_params <- NULL

  for(i in 1:nrow(params)) {
    # Extract parameters
    param <- list(
      objective = "binary:logistic",
      max_depth = params$max_depth[i],
      eta = params$eta[i],
      gamma = params$gamma[i]
    )

    # Train model
    model <- xgboost(params = param, data = dtrain, nrounds = params$nrounds[i], verbose = 0)
    
    # Predict on test data
    preds_prob <- predict(model, dtest)

    # Apply optimal threshold
    roc_obj <- roc(actual, preds_prob)
    optimal_threshold <- coords(roc_obj, "best", ret="threshold")
    preds_binary_optimal <- ifelse(preds_prob > as.numeric(optimal_threshold), 1, 0)
    
    # Compute confusion matrix and sensitivity
    cm <- table(actual, preds_binary_optimal)
    sensitivity <- cm[2, 2] / sum(cm[2, ])

    if (sensitivity > best_sensitivity) {
      best_sensitivity <- sensitivity
      best_params <- param
      best_params$nrounds <- params$nrounds[i]
    }
  }
  list(best_params = best_params, best_sensitivity = best_sensitivity)
}

# Perform grid search
folds <- 5
best_model <- grid_search_sensitivity(grid, trainData_matrix, testData_matrix, testData$poor_20, folds)

# Final Model Training with Best Parameters
final_params <- best_model$best_params

final_model <- xgboost(params = final_params, data = trainData_matrix, 
                       nrounds = final_params$nrounds, verbose = 0)

# Predict on test data with final model
final_preds <- predict(final_model, testData_matrix)

# ROC and AUC for final model
roc_obj_final <- roc(testData$poor_20, final_preds)
auc_value_final <- auc(roc_obj_final)

# Apply the optimal threshold
final_preds_binary <- ifelse(final_preds > as.numeric(optimal_threshold), 1, 0)

# Confusion Matrix for final model
cm_final <- table(testData$poor_20, final_preds_binary)

# Evaluation Metrics Calculation
accuracy_final <- sum(diag(cm_final)) / sum(cm_final)
precision_final <- cm_final[2, 2] / sum(cm_final[, 2])
sensitivity_final <- cm_final[2, 2] / sum(cm_final[2, ])
f1_score_final <- 2 * ((precision_final * sensitivity_final) / (precision_final + sensitivity_final))

# Create a data frame to store the metrics
metrics_table_final <- data.frame(Accuracy = accuracy_final, 
                                    Precision = precision_final, 
                                    sensitivity = sensitivity_final, 
                                    F1_Score = f1_score_final, 
                                    AUC = auc_value_final)

# Print the metrics table with Optimal Threshold
print(metrics_table_final)
xtable(metrics_table_final)
print(cm_final)
xtable(cm_final)

############ END OF SCRIPT ###########


## TO DO
#feature importance

#cross validation

