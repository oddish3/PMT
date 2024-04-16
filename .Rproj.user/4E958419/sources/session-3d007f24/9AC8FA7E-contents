#----
# A Poor Means Test Replication Study
#Regressions
#Author : Sol Yates

#wd 
rm(list=ls())
# setwd("/home/oddish3/Documents/R_folder/UG/PMT/Scripts")
#----
library(dplyr)
library(tidyverse)
library(haven)
library(modelr)
library(broom)
library(lmtest)  # For clustering standard errors
library(stargazer)  # For exporting results
library(sandwich)
library(magrittr)
#----

#MAIN FILE
data <- read_dta("data/out/PMT_CLEAN_SY.dta")

#PMT REGRESSIONS
# Defining the variables

###   basic PMT
wealth <- c("toilet_flush", "toilet_pit", "floor_finish", "wall_finish", "roof_finish", 
            "fuel_elecgas", "fuel_charcoal")
hh <- c("urban", "female_head", "edu_head_primary", "edu_head_secondary", "div_sep_head", 
        "widow_head", "work_paid_head", "work_selfemp_nonf_head", "muslim", "christian", 
        "share_05f", "share_05m", "share_614f", "share_614m", "share_65plusf", "share_65plusm")

## basic PMT regression and prediction 

data$yhat_pmt_sh_consump <- NA
models <- list()
model_summaries<- list()

for (x in unique(data$country)) {
  df <- subset(data, country == x)
  
  # Construct the model formula
  formula_str <- paste("consump ~", paste(c(wealth, hh), collapse = " + "),
                       "+ as.factor(hhsize_cat) + as.factor(age_head_cat) + as.factor(state) + as.numeric(month)")
  formula <- as.formula(formula_str)
  
  # Fit the linear model with clustered standard errors
  model <- lm(formula, data = df)
  model_clust <- coeftest(model, vcov = vcovCL(model, cluster = df$EA))
  
  # Predict and update values for the current country
  yhat <- predict(model, newdata = df)
  data$yhat_pmt_sh_consump[data$country == x] <- yhat
  
  # Store the model and the standard errors
  models[[x]] <- model
  model_summaries[[x]] <- broom::tidy(model_clust)
}

combined_results <- bind_rows(model_summaries, .id = "country") #object matching table a2 needs some corrections
write_dta(data, "data/out/PMT_CLEAN_SY1.dta")

## Poor (binary) indicators as dependent variable basic PMT
# OLS and PROBIT 

## OLS 

# Define the dependent variables
depvars <- c("poor_20", "poor_40")

# Loop through each dependent variable
for (x in depvars) {
  # Generate new variables
  data[[paste0("yhat_pmt_sh_", x)]] <- NA
  data[[paste0("yhat_pmt_sh_pr_", x)]] <- NA
}

models <- list()
model_summaries <- list()
predictions <- list()

# # Loop through each country
# for (x in unique(data$country)) {
#   df <- subset(data, country == x)
#   
#   # Loop through each dependent variable
#   for (var in depvars) {
#     # Construct the model formula
#     formula_str <- paste(var, "~", paste(c(wealth, hh), collapse = " + "),
#                          "+ as.factor(hhsize_cat) + as.factor(age_head_cat) + as.factor(state) + as.numeric(month)")
#     formula <- as.formula(formula_str)
#     
#     # Fit the linear model with clustered standard errors
#     model <- lm(formula, data = df)
#     model_clust <- coeftest(model, vcov = vcovCL(model, cluster = df$EA))
#     
#     # Predict values for the current country and variable
#     yhat <- predict(model, newdata = df)
#     
#     # Store predictions in a list
#     predictions[[paste0("yhat_pmt_sh_", var, "_", x)]] <- yhat
#     
#     # Store the model and the standard errors
#     model_name <- paste0(x, "_", var)
#     models[[model_name]] <- model
#     model_summaries[[model_name]] <- broom::tidy(model_clust)
#   }
# }
# 
# # Combine all model summaries
# combined_results <- bind_rows(model_summaries, .id = "model")
# 
# # Add predictions back to the data dataframe
# for (name in names(predictions)) {
#   data[[name]] <- predictions[[name]]
# }
data %<>% filter(country == "Uganda") 
write_dta(data, "data/out/PMT_CLEAN_SY2.dta")

