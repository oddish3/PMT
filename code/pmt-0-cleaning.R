#----
# A Poor Means Test Replication Study
#Cleaning
#Author : Sol Yates

#wd 
rm(list=ls())
# setwd("/home/oddish3/Documents/R_folder/UG/PMT/Scripts")


#----
library(dplyr)
library(tidyverse)
library(haven)
library(janitor)
library(statar)

#----
#MAIN FILE
data <- read_dta("data/in/Brown_Ravallion_vandeWalle_PMT.dta")

## creating household size category (shld be ok)
data$hhsize_cat <- cut(data$hhsize, breaks = c(0, 1, 3, 5, 7, 9, Inf), right = FALSE, labels = FALSE)
data$hhsize_cat[data$hhsize >= 9] <- 9

## adjusting month variable for specific countries (ok, but what happen to intercept?)
data$month <- ifelse(data$country %in% c("BurkinaFaso", "Ethiopia", "Malawi", "Mali"), 1, data$month)

## Generating and Adjusting Population Weights (ok) 
data$popweight1 <- with(data, ifelse(is.na(hhweight*hhsize) | hhweight*hhsize == 0, popweight, hhweight*hhsize))

## calculating household weight for nigeria (ok)
data$hhweight <- ifelse(data$country == "Nigeria", data$popweight / data$hhsize, data$hhweight)

#summarising population weights and household weights by country (ok)
summary_data <- data %>%
  group_by(country) %>%
  summarise(across(c(popweight, popweight1, hhweight), sum, na.rm = TRUE))

#dropping observations with missing weights (ok) 
data <- data %>%
  filter(!is.na(hhweight) & !is.na(popweight1)) #correct 1335 dropped

#handling missing values in various share variables (ok) 
share_vars <- c("share_05f", "share_05m", "share_614f", "share_614m", "share_65plusf", "share_65plusm", "share_widow", "share_disabledm", "share_disabledf", "share_orphanm", "share_orphanf")
for(var in share_vars) {
  data[[var]] <- ifelse(is.na(data[[var]]), 0, data[[var]])
}

## table 2
## calculating and labelling variables related to maen consumption (ok) 
data <- data %>%
  group_by(country) %>%
  mutate(mean_consump = weighted.mean(real_consumption_pc, w = popweight1, na.rm = TRUE)) %>%
  ungroup()

#Create the 'below_mean' variable and set it to 0 initially
data$below_mean <- 0

#Replace 'below_mean' with 1 for rows where 'real_consumption_pc' is below 'mean_consump'
data$below_mean[data$real_consumption_pc <= data$mean_consump] <- 1

#Calculate the weighted percentage of 'below_mean' by 'country'
data %>%
  group_by(country) %>%
  summarise(weighted_below_mean = sprintf("%.3f", sum(below_mean * popweight1) / sum(popweight1)))

## dropping obs based on missing values in certain variables (ok)
vars <- c("water_piped", "water_well", "toilet_flush", "toilet_pit", "floor_finish", 
          "wall_finish", "roof_finish", "members_per_room", "kitchen_room", "fuel_elecgas", 
          "fuel_charcoal", "electric", "radio", "telev", "fridge", "bicycle", "motorbike", 
          "car", "telephone", "mobile_phone", "computer", "video", "stove_any", "sew_machine", 
          "aircon", "iron", "satelite", "generator", "own_house", "urban", "female_head", 
          "edu_head_primary", "edu_head_secondary", "max_edu_primary", "max_edu_secondary", 
          "div_sep_head", "widow_head", "nevermar_head", "work_paid_head", "work_selfemp_nonf_head", 
          "work_daily_head", "share_05f", "share_05m", "share_614f", "share_614m", 
          "share_65plusf", "share_65plusm", "muslim", "christian", "month", "age_head_cat", 
          "hhsize_cat")

for(var in vars) {
  data$real_consumption_pc <- ifelse(is.na(data[[var]]), NA, data$real_consumption_pc)
}

data <- data[!is.na(data$real_consumption_pc), ] # CORRECT 1942 OBS DROPPED

## generating log of real consumption per capita  (ok) 
data$consump <- log(data$real_consumption_pc)

#tabulating log real consumption per capita per country (ok) 
data %>%
  group_by(country) %>%
  summarise(
    mean = mean(consump, na.rm = TRUE),
    median = median(consump, na.rm = TRUE),
    min = min(consump, na.rm = TRUE),
    max = max(consump, na.rm = TRUE),
    sd = sd(consump, na.rm = TRUE)
  )

## generating consumption percentiles by country (ok) 
countries <- c("BurkinaFaso", "Ethiopia", "Ghana", "Malawi", "Mali", "Niger", "Nigeria", "Tanzania", "Uganda")

# Assuming 'data' is your data frame and 'country' is a vector of country names

# Initialize the 'consumption_pctile' column with NA
data$consumption_pctile <- NA

# Loop over each country name
for (x in countries) {
  # Subsetting data and exclude missing values of consump
  df_x <- data[data$country == x & !is.na(data$consump), ]
  
  # Compute weighted quantiles using xtile function or a similar R function
  # The xtile function should be defined or sourced from an appropriate package
  quantiles_x <- xtile(df_x$consump, n = 100, wt = df_x$hhweight)
  
  # Assign the calculated percentiles back to the main dataset
  data$consumption_pctile[data$country == x] <- quantiles_x
}


## table a1 in appendix
# Defining the variables
wealth_vars <- c("real_consumption_pc", "water_piped", "water_well", "toilet_flush", "toilet_pit", "floor_natural", "floor_rudiment", "floor_finish", "wall_natural", "wall_rudiment", "wall_finish", "roof_natural", "roof_rudiment", "roof_finish", "members_per_room", "kitchen_room", "fuel_elecgas", "fuel_charcoal", "fuel_wood")
assets_vars <- c("electric", "radio", "telev", "fridge", "bicycle", "motorbike", "car", "telephone", "mobile_phone", "computer", "video", "stove_any", "sew_machine", "aircon", "iron", "satelite", "generator", "own_house")
hh_vars <- c("urban", "age_head", "female_head", "edu_head_primary", "edu_head_secondary", "max_edu_primary", "max_edu_secondary", "ever_married_head", "married_head", "div_sep_head", "widow_head", "nevermar_head", "hhsize", "work_paid_head", "work_selfemp_nonf_head", "work_selfemp_farm_head", "work_daily_head", "muslim", "christian", "share_05f", "share_05m", "share_614f", "share_614m", "share_1564f", "share_1564m", "share_65plusf", "share_65plusm", "share_widow", "share_disabledm", "share_disabledf", "share_orphanm", "share_orphanf")

# Function to calculate weighted mean for a vector
weighted_mean <- function(x, w) {
  sum(x * w, na.rm = TRUE) / sum(w, na.rm = TRUE)
}

# Applying the function to each set of variables
calculate_stats <- function(data, vars) {
  data %>%
    select(country, hhweight, all_of(vars)) %>%
    group_by(country) %>%
    summarise(across(all_of(vars), ~ weighted_mean(., hhweight))) %>%
    ungroup()
}

# Calculating stats for each group of variables
wealth_stats <- calculate_stats(data, wealth_vars)
assets_stats <- calculate_stats(data, assets_vars)
hh_stats <- calculate_stats(data, hh_vars)


## generating pov lines ## # there are some slight discrepancies from how you do this

data$pov <-ifelse(data$consumption_pctile == 20, data$consump, 0)
data <- data %>%
  group_by(country) %>%
  mutate(pov_line_20 = max(pov, na.rm = TRUE)) %>%
  ungroup() %>%
  select(-pov)

data$pov <-ifelse(data$consumption_pctile == 40, data$consump, 0)
data <- data %>%
  group_by(country) %>%
  mutate(pov_line_40 = max(pov, na.rm = TRUE)) %>%
  ungroup() %>%
  select(-pov)

#generate poverty rates pre transfers 
data$poor_20 <- ifelse(data$consump <= data$pov_line_20, 1, 0)
data$poor_40 <- ifelse(data$consump <= data$pov_line_40, 1, 0)

#checking
data %>% group_by(country) %>% 
  summarise(weighted_mean_40 = weighted.mean(poor_40, hhweight, na.rm = TRUE),
            weighted_mean_20 = weighted.mean(poor_20, hhweight, na.rm = TRUE))


write_dta(data, "data/out/PMT_CLEAN_SY.dta")

