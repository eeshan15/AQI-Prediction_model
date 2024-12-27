# AQI Prediction Model

This project focuses on predicting the Air Quality Index (AQI) using various machine learning techniques. The main goal is to develop a robust model that accurately predicts AQI values based on historical data, helping users understand air quality patterns and take necessary precautions.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Air quality has a significant impact on public health, and predicting AQI can help communities prepare for hazardous conditions. This project implements a machine learning pipeline to predict AQI using various data preprocessing and modeling techniques.

## Features

- Data preprocessing and handling missing values using advanced imputation techniques.
- Data visualization using Matplotlib and Seaborn.
- Implementation of multiple machine learning models, including Random Forest, Gradient Boosting, and XGBoost.
- Selection of the best performing model based on evaluation metrics.

## Data Preprocessing

The data preprocessing steps include:

- **Handling Missing Values:** Missing values in the dataset were filled using the MICE (Multiple Imputation by Chained Equations) technique. This was implemented using the Iterative Imputer from scikit-learn with Ridge, Elastic Net, and Bayesian Regression as estimators.
  
- **Data Visualization:** Data insights and visual patterns were explored using Pandas, Numpy, Matplotlib, and Seaborn.

## Machine Learning Models

Several machine learning models were tested to find the best performing one:

- **Random Forest (Selected Model):** Performed best among all models.
- **XGBoost**
- **Gradient Boosting**

The Random Forest model was selected based on evaluation metrics such as accuracy, mean squared error, and R² score.

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/KavyaSoni123/AQI-predictor-model.git
cd aqi-prediction-model
pip install -r requirements.txt
