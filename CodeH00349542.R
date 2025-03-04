# ===================================================================
# Author: Lih Woei Siow
# Date: 20 Nov 2024
# Purpose: This script is designed to address the Machine Learning 
#          for Risk and Insurance project (F71RA 2024-25). It includes
#          data preprocessing, exploratory data analysis, PCA, linear
#          regression, and neural network modeling for insurance claims.
# ===================================================================

# ==========================
# R Version: This script was written and tested in R version 4.3.0.
# ==========================

# Clear the environment to avoid conflicts with previous variables or objects
rm(list=ls())

# Load necessary libraries for the analysis
library(tidyverse) #1.3.1 (for data manipulation)
library(keras) #2.8.0 (for deep learning)
library(reticulate) #1.25 (for integrating Python environments)
library(ggplot2) #3.3.5 (for visualizations)
library(cluster) #2.1.3 (for clustering analysis)
library(ClusterR) #1.2.5 (for clustering algorithms)
library(readxl) #1.4.0 (for reading Excel files)
library(dplyr) #1.0.7 (for data manipulation)
library(magrittr) #2.0.2 (for functional piping)
library(ggbiplot) #0.55 (for PCA visualization)
library(ggcorrplot) #0.1.3 (for correlation heatmaps)
library(GGally) #2.1.2 (for data exploration)
library(MASS) # (for statistical analysis)

# Set the working directory to the path containing data and scripts
setwd("/Users/siowlihwoei/Desktop/Programming/MSc/Sem 1/Coursework")

# Use a specified Conda environment for TensorFlow (Python backend for Keras)
use_condaenv("tf-env", required = TRUE)
py_config() # Verify the Python environment configuration

# ==========================
# Part 2: Clean and preprocess motor insurance data for analysis.
# ==========================

# Function to load and summarize the motor insurance dataset
motor_insurance <- read_xlsx('motor_insurance.xlsx')

# Part 2(a): Data Structure Overview
# ----------------------------------------------------
str(motor_insurance) # Display the structure of the dataset
summary(motor_insurance) # Provide an overview of the dataset

# Part 2(b): Analyze 'loss' Column
# ----------------------------------------------------
loss = motor_insurance$loss # Extract the 'loss' column for further analysis

# Calculate Range and Interquartile Range (IQR) for the 'loss' column
range_loss <- range(loss, na.rm = TRUE) ; range_loss
iqr_loss <- IQR(loss, na.rm = TRUE) ; iqr_loss
percentiles <- quantile(loss, probs = c(0.005, 0.995), na.rm = TRUE) ; percentiles

# Set up the plotting area to display two boxplots side-by-side
par(mfrow = c(1, 2)) # Create a plotting layout with 1 row and 2 columns

# Plot a boxplot to identify outliers before data cleaning
boxplot(loss, main = "Boxplot of the Total Loss Incurred to the Claimant", ylab = "Loss")

# Replace outliers in the 'loss' column with NA
# - Identify values below the 0.5th percentile or above the 99.5th percentile
# - Replace these extreme values with NA to handle outliers effectively
motor_insurance = motor_insurance %>% mutate(loss = ifelse(
  loss < percentiles[1] | loss > percentiles[2], NA, loss))

# Update the 'loss' variable to reflect the modified data in the dataset
loss = motor_insurance$loss

# Plot a boxplot again to visualize the 'loss' column after removal of outliers
boxplot(loss, main = "Boxplot of the Total Loss Incurred to the Claimant", ylab = "Loss")

# Part 2(c): Identify and Replace Outliers in the 'loss' Column
# ----------------------------------------------------
#Filter Out Outliers and Negatives
motor_insurance <- motor_insurance %>% filter(!is.na(loss) & loss >= 0)

# Part 2(d): Box-Cox Transformation to Reduce Skewness
# ----------------------------------------------------
# Apply Box-Cox transformation to identify the optimal lambda for transforming the 'loss' column
# - Fit a linear model with just the intercept (essentially to transform the 'loss' distribution)
boxcox_loss <- boxcox(lm(loss ~ 1, data = motor_insurance))

# Extract the optimal lambda value from the Box-Cox transformation results
optimal_lambda <- boxcox_loss$x[which.max(boxcox_loss$y)] ; optimal_lambda # The lambda value that maximizes the likelihood

# Transform the 'loss' column using the optimal lambda to stabilize variance and reduce skewness
motor_insurance <- motor_insurance %>% mutate(transformedLoss = (loss^optimal_lambda - 1) / optimal_lambda)

# Set up the plotting area to display two histograms side-by-side
par(mfrow = c(1, 2))  #Set plotting area to have 1 row and 2 columns for comparison

# Plot histogram of the original 'loss' variable with a normal curve overlay
hist(motor_insurance$loss, 
     main = "Original Loss Distribution with Skewed Curve", 
     xlab = "Loss", 
     col = "lightblue", 
     freq = FALSE, 
     xlim = c(0, 250))  # Set x-axis limit to focus on skewed part

# Add a normal distribution curve to the original 'loss' histogram
curve(dnorm(x, mean = mean(motor_insurance$loss, na.rm = TRUE), 
            sd = sd(motor_insurance$loss, na.rm = TRUE)), 
      add = TRUE, col = "blue", lwd = 2)

# Plot histogram of the transformed 'loss' variable with a normal curve overlay
hist(motor_insurance$transformedLoss, 
     main = "Transformed Loss Distribution with Normal Curve", 
     xlab = "Transformed Loss", 
     col = "lightgreen", 
     freq = FALSE)

# Add a normal distribution curve to the transformed 'loss' histogram
curve(dnorm(x, mean = mean(motor_insurance$transformedLoss, na.rm = TRUE), 
            sd = sd(motor_insurance$transformedLoss, na.rm = TRUE)), 
      add = TRUE, col = "blue", lwd = 2)

# Part 2(e): Mean Claims by Gender
# ----------------------------------------------------
# Filter out rows with missing values in 'claimantGender' before performing group analysis
motor_insurance_gender = motor_insurance %>% filter(!is.na(claimantGender))

# Calculate mean claims by gender (1 for Male, 2 for Female)
mean_claims <- motor_insurance_gender %>%
  group_by(motor_insurance_gender$claimantGender) %>%
  summarize(mean_claims = mean(loss, na.rm = TRUE)) ; mean_claims

# Part 2(f): Age Group Categorization
# -----------------------------------
# Filter out rows with missing values in 'claimantAge' before categorizing into age groups
motor_insurance = motor_insurance %>% filter(!is.na(claimantAge))

# Create a new categorical variable 'ageGroup' based on 'claimantAge' ranges
motor_insurance <- motor_insurance %>%
  mutate(ageGroup = cut(claimantAge,
                         breaks = c(-Inf, 25, 35, 42, 72, Inf),
                         labels = c("under 25", "26-35", "36-42", "43-72", "72 or above"))) ; motor_insurance

# Calculate the number of observations and mean loss for each age group
motor_insurance %>% group_by(ageGroup) %>% summarise(count=n(),mean(loss))

# ==========================
# - Part 3: Perform Principal Component Analysis and linear regression modeling.
# ==========================

#Load the Data
marine <- read_xlsx('Marine.xlsx')  # Load the Marine dataset from an Excel file

# 3(a): Calculate the Correlation Matrix
# --------------------------------------
# Calculate the correlation matrix of the dataset to understand relationships between variables
correlation_matrix <- cor(marine) ; correlation_matrix

# Convert the correlation matrix to long format and plot using ggplot
# This helps visualize the relationships between different variables in the dataset
ggcorrplot(correlation_matrix,
           lab = TRUE,
           lab_size = 3,
           tl.cex = 9,
           title = "Correlation Heatmap of Marine Dataset Variables")

# Remove duplicate and diagonal entries by setting the lower triangle and diagonal to NA
correlation_matrix[lower.tri(correlation_matrix, diag = TRUE)] <- NA

# Find the indices of the most correlated pair (excluding NA)
max_corr <- which(abs(correlation_matrix) == max(abs(correlation_matrix), na.rm = TRUE), arr.ind = TRUE)

# Extract and print the names of the most correlated variables
most_correlated <- c(rownames(correlation_matrix)[max_corr[1]],
                     colnames(correlation_matrix)[max_corr[2]])
cat("The two most correlated variables are:", most_correlated[1], "and", most_correlated[2])

# 3(b): Scatter Plot Matrix and Q-Q Plots
# ---------------------------------------
# Scatter Plot Matrix using GGally
par(mfrow = c(1,1))
ggpairs(marine, title = "Scatter Plot Matrix of Marine Dataset")

# Generate Q-Q plots for each variable to check normality
par(mfrow = c(1,ncol(marine))) # Set up plotting area for Q-Q plots of each variable
for(i in 1:ncol(marine)) {
  var <- as.numeric(marine[[i]])  # Convert each column to a numeric vector
  qqnorm(var, main = paste0('Q-Q Plot of ', colnames(marine)[i]))
  qqline(var, col = 2, lty = 2, lwd = 2)
}

# Standardize data using Z-score standardization
marine_std <- as.data.frame(scale(marine))

# Perform Principal Component Analysis (PCA)
# ------------------------------------------
# Select all columns except 'claimCount' and 'claimCost' for PCA
marine_data <- marine_std %>% dplyr::select(!claimCount & !claimCost)

pca_result <- princomp(marine_data, cor = TRUE) # Conduct PCA on the standardized dataset
summary(pca_result) # Summary of PCA showing importance of each principal component
pca_result$loadings[,1]  # Show loadings for the first principal component

# Calculate variance explained by each component and plot Scree Plot
variance_explained <- pca_result$sdev^2 # Compute variance explained by each component
proportion_variance <- variance_explained / sum(variance_explained) # Calculate proportion of variance
cumulative_variance <- cumsum(proportion_variance) # Cumulative variance explained

# Scree plot with elbow method visualization
par(mfrow = c(1, 1))
plot(proportion_variance, type = "b", pch = 19, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", main = "Scree Plot")
abline(v = which.max(cumulative_variance >= 0.80), col = "red", lty = 2)  # Marks 80% variance threshold

# Determine the minimum number of components that retain at least 80% of the variance
num_comp <- which(cumulative_variance >= 0.80)[1]
cat("The number of components retaining at least 80% of variation is:", num_comp)

# 3(c): Plot PCA Biplot
# ---------------------
#  Create PCA biplot to visualize components
biplot(pca_result, scale = 0, main = "PCA Biplot", cex = 0.5, ylim=c(-4,4))

# 3(d): Multiple Linear Regression with Principal Components
# ----------------------------------------------------------
# Select the principal components that explain at least 80% of variance
pca_select <- pca_result$scores[,1:num_comp]

# Combine the selected principal components with the standardized 'claimCost' variable
marine_data_pca <- data.frame(pca_select, claimCost = marine_std$claimCost)

# Fit a multiple linear regression model to predict 'claimCost' using selected principal components
pca_lm <- lm(claimCost ~ ., data = marine_data_pca)
# Display summary statistics for the linear regression model
summary(pca_lm)

# 3(e): Predicting Claim Costs for New Observations
# -------------------------------------------------
# Create new data for prediction with standardized features
new_spec = data.frame(vesselVal = 22, vesselAge = 14.5, distance = 25, duration = 10)

# Standardize the new data based on the original dataset means and standard deviations
new_spec$vesselVal = (new_spec$vesselVal - mean(marine$vesselVal))/sd(marine$vesselVal)
new_spec$vesselAge = (new_spec$vesselAge - mean(marine$vesselAge))/sd(marine$vesselAge)
new_spec$distance = (new_spec$distance - mean(marine$distance))/sd(marine$distance)
new_spec$duration = (new_spec$duration - mean(marine$duration))/sd(marine$duration)

# Apply the PCA transformation to the new standardized data
new_spec_pca = as.data.frame(predict(pca_result, new_spec))

# Predict 'claimCost' for the new data using the fitted regression model
prediction = predict(pca_lm,new_spec_pca); prediction

# Reverse the standardization for the predicted 'claimCost' to obtain an interpretable value
unstand_prediction = prediction * sd(marine$claimCost) + mean(marine$claimCost); unstand_prediction

# ==========================
# - Part 4: Build and evaluate a deep neural network for predicting insurance claims.
# ==========================

# Load your dataset here
data <- read_csv("freMTPL2freq.csv") # Load the dataset from a CSV file

# 4(a): Data Preparation
# ----------------------
# Display the structure of the dataset to understand variable types and dimensions
str(data)

# Convert character columns to factors for easier modeling
for(i in seq_along(data)){
  if(is.character(data[[i]])){
    data[[i]] <- factor(data[[i]])
  }
}

# Cap 'ClaimNb' values to a maximum of 4
data$ClaimNb <- pmin(data$ClaimNb, 4) # Prevent extreme values in 'ClaimNb' by capping it at 4

# Cap 'Exposure' values to a maximum of 1
data$Exposure <- pmin(data$Exposure, 1) # Cap 'Exposure' at 1 to normalize this feature

# Split Data into Learning and Test Sets
learn_idx <- sample(1:nrow(data), round(0.9*nrow(data)), replace = FALSE) # Randomly select 90% of data for training
learn <- data[learn_idx, ]  # Training data
test <- data[-learn_idx, ]  # Test data
n_l <- nrow(learn) # Number of rows in training set
n_t <- nrow(test) # Number of rows in test set

# Define function to perform Min-Max scaling (normalization to range [-1, 1])
MM_scaling <- function(x){ 2*(x-min(x))/(max(x)-min(x)) - 1}

# Create a scaled dataset specifically for neural network modeling
data_NN <- data.frame(ClaimNb = data$ClaimNb) # Initialize data frame with 'ClaimNb' column
data_NN$DriveAge <- MM_scaling(data$DrivAge) # Scale 'DrivAge'
data_NN$BonusMalus <- MM_scaling(data$BonusMalus) # Scale 'BonusMalus'
data_NN$Area <- MM_scaling(as.integer(data$Area)) # Scale 'Area' (convert to integer first)
data_NN$VehPower <- MM_scaling(as.numeric(data$VehPower)) # Scale 'VehPower'
data_NN$VehAge <- MM_scaling(as.numeric(data$VehAge)) # Scale 'VehAge'
data_NN$Density <- MM_scaling(data$Density) # Scale 'Density'
data_NN$VehGas <- MM_scaling(as.integer(data$VehGas)) # Scale 'VehGas'

# Split the scaled data into learning and test sets
learn_NN <- data_NN[learn_idx,] # Training data for neural network
test_NN <- data_NN[-learn_idx,] # Test data for neural network

# Create matrices for model input (excluding 'ClaimNb' which is the target variable)
Design_learn <- as.matrix(learn_NN[ , -1])
Design_test <- as.matrix(test_NN[ ,-1])

# Extract categorical variables for embedding layers
Br_learn <- as.matrix(as.integer(learn$VehBrand)) - 1 # Vehicle brand for training set (zero-indexed)
Br_test <- as.matrix(as.integer(test$VehBrand)) - 1 # Vehicle brand for test set

Re_learn <- as.matrix(as.integer(learn$Region)) - 1 # Region for training set (zero-indexed)
Re_test <- as.matrix(as.integer(test$Region)) - 1 # Region for test set

# Extract Exposure data and apply log transformation for use as an offset
Vol_learn <- as.matrix(learn$Exposure)
Vol_test <- as.matrix(test$Exposure)
LogVol_learn <- log(Vol_learn) # Log transformation of 'Exposure' for training
LogVol_test <- log(Vol_test)  # Log transformation of 'Exposure' for test

# Set the target variable for training and test sets
Y_learn <- as.matrix(learn_NN$ClaimNb)
Y_test <- as.matrix(test_NN$ClaimNb)

# 4(b): Define the Neural Network Architecture
# --------------------------------------------
# Define layer sizes and embedding dimensions
q1 <- 27 # the dimension of the 1st hidden layer
q2 <- 22 # the dimension of the 2nd hidden layer
q3 <- 17 # the dimension of the 3rd hidden layer
qEmb <- 2 # the dimension of the embedded layer for "VehBrand" and "Region"

epochs <- 200 # number of epochs to train the model
batchsize <- 10000 # number of samples per gradient update

# Step 2.2: Define Input Layers
# -----------------------------
# Input layer for continuous features
Design <- layer_input(shape = ncol(learn_NN) - 1, dtype = 'float32', name = 'Design')

# Input layers for categorical features
Br_ndistinct <- length(unique(learn$VehBrand)) # Number of unique vehicle brands = 11
Re_ndistinct <- length(unique(learn$Region)) # Number of unique regions = 21

VehBrand <- layer_input(shape = 1, dtype = 'int32', name = 'VehBrand') # Input layer for 'VehBrand'
Region <- layer_input(shape = 1, dtype = 'int32', name = 'Region')  # Input layer for 'Region'

# Input layer for Exposure (as the offset)
LogVol <- layer_input(shape = 1, dtype = 'float32', name = 'LogVol')
Vol <- layer_input(shape = 1, dtype = 'float32', name = 'Vol')

# Step 2.3: Embedding Layers for Categorical Features
# ---------------------------------------------------
# Embedding layer for 'VehBrand'
BrEmb = VehBrand %>%
  layer_embedding(input_dim = Br_ndistinct, output_dim = qEmb, input_length = 1, name = 'BrEmb') %>%
  layer_flatten(name = 'Br_flat')

# Embedding layer for 'Region'
ReEmb = Region %>%
  layer_embedding(input_dim = Re_ndistinct, output_dim = qEmb, input_length = 1, name = 'ReEmb') %>%
  layer_flatten(name = 'Re_flat')


# Step 2.4: Main Neural Network Architecture and Output Layer
# -----------------------------------------------------------
# Concatenate the input layers and build the hidden layers
Network <- list(Design, BrEmb, ReEmb) %>% layer_concatenate(name = 'concate') %>%
  layer_dense(units = q1, activation = 'tanh', name = 'hidden1') %>%  # 1st hidden layer
  layer_dense(units = q2, activation = 'tanh', name = 'hidden2') %>%  # 2nd hidden layer
  layer_dense(units = q3, activation = 'tanh', name = 'hidden3') %>%  # 3rd hidden layer
  layer_dense(units = 1, activation = 'linear', name = 'Network')  # provide one neuron in the output layer

# Define the output layer for the response variable using the offset layer
Response = (Network + LogVol) %>%
  # give the response
  layer_dense(units = 1,
              activation = 'exponential',
              name = 'Response',
              trainable = FALSE,
              weights = list(array(1, dim = c(1,1)), array(0, dim = c(1))))


# Step 2.5: Model Configuration and Fitting
# -----------------------------------------
# Assemble the model with input and output layers
model <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))
summary(model) # Display model summary to show architecture details

# Compile the model with loss function and optimizer
model %>% compile(
  loss = 'poisson', # set poisson deviance loss function as the objective loss function
  optimizer = 'nadam' # Optimizer used for training
)

# Model fitting by running gradient descent method to minimize the objective loss function
{ 
  t1 <- proc.time() # Start timer to measure training time
  
  fit <- model %>% fit(
    list(Design_learn, Br_learn, Re_learn, LogVol_learn), # all predictors
    Y_learn, # response variable
    verbose = 1, # Verbose = 1 for detailed training progress
    epochs = epochs, # Number of training epochs = 200
    batch_size = batchsize, # Batch size for each iteration = 10,000
    validation_split = 0.2 # Use 20% of data as validation set
  )
  
  print(proc.time()-t1) # Print elapsed training time
}

# Predict 'ClaimNb' for both training and test datasets
learn$nn0 <- as.vector(model %>% predict(list(Design_learn, Br_learn, Re_learn, LogVol_learn)))
test$nn0 <- as.vector(model %>% predict(list(Design_test, Br_test, Re_test, LogVol_test)))

# Define custom function to compute deviance loss
dev.loss <- function(y, mu, density.func) {
  logL.tilde <- log(density.func(y, y))
  logL.hat <- log(density.func(y, mu))
  2 * mean(logL.tilde - logL.hat)
}

# Compute deviance loss for the training data predictions
dev.loss(y = learn$ClaimNb, mu = learn$nn0, dpois)

