


![banner](/images/LHL_banner.png)

Welcome to the repository for the Supervised Learning Midterm project. This project focuses on building and evaluating various machine learning regression models to predict house prices based on a given dataset.

## Table of Contents

- [Overview](#overview)
- [Project Steps](#project-steps)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Selection and Training](#model-selection-and-training)
  - [Results](#results)
  - [Challenges](#challenges)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to explore and implement different supervised learning regression models to predict the price of houses based on various features such as square footage, number of bedrooms, number of bathrooms, and other relevant attributes. We evaluate the performance of these models using various metrics to determine the best model for this task.

## Project Steps

### Data Cleaning

- **Loading the Data**: Loaded data from multiple json files and combined into one dataframe
- **Handling Missing Values**: Imputated and removed null values
- **Encoding Categorical Variables**: Categorical variables are encoded using techniques such as one-hot encoding.

### Exploratory Data Analysis (EDA)

- **Descriptive Statistics**: say some stuff here
- **Visualizations**: Various plots such as histograms, boxplots, and scatter plots are created to understand the distributions and relationships between features.
- **Correlation Analysis**: Correlation heatmaps are generated to identify relationships between features and the target variable.

### Model Selection and Training

- **Linear Regression**: Basic linear regression model.
- **Ridge Regression**: Linear regression with L2 regularization.
- **Lasso Regression**: Linear regression with L1 regularization.
- **XGBoost Regressor**: Gradient boosting regression model.

### Results

#### Best Performing Model: Decision Tree Regressor

- **Mean Squared Error (MSE)**: 495,867,768.60
- **Mean Absolute Error (MAE)**: 2,479.34
- **R² Score**: 0.9985

#### Comparison with Other Models

| Model                         | Mean Squared Error (MSE)  | Mean Absolute Error (MAE)  | R² Score  |
|-------------------------------|---------------------------|----------------------------|-----------|
| **Decision Tree**             | 495,867,768.60            | 2,479.34                   | 0.9985    |
| **Random Forest**             | 1,024,435,105.73          | 11,535.48                  | 0.9969    |
| **XGBoost**                   | 2,504,522,904.09          | 30,133.23                  | 0.9924    |
| **Linear Regression**         | 185,504,913,290.17        | 183,278.21                 | 0.4392    |
| **Support Vector Machine**    | 346,285,720,808.71        | 220,627.45                 | -0.0468   |
| **K-Nearest Neighbors**       | 26,662,809,917.36         | 70,867.77                  | 0.9194    |
| **Gradient Boosting**         | 24,843,589,621.09         | 105,160.04                 | 0.9249    |
| **AdaBoost**                  | 105,503,589,057.43        | 281,912.39                 | 0.6811    |
| **Ridge Regression**          | 185,530,659,977.92        | 183,475.83                 | 0.4391    |
| **Lasso Regression**          | 185,503,888,825.00        | 183,296.20                 | 0.4392    |
| **ElasticNet Regression**     | 234,388,902,656.75        | 189,973.07                 | 0.2914    |

### Challenges

- **Handling Missing Values**: Dealing with missing data and choosing appropriate imputation strategies.
- **Categorical Encoding**: Ensuring that categorical variables are properly encoded for model training.
- **Model Tuning**: Selecting the right hyperparameters for models like Ridge, Lasso, and XGBoost.
- **Evaluation Metrics**: Choosing the right metrics to evaluate model performance and interpret results effectively.

## Directory Structure

The repository is organized as follows:

```plaintext
LHL_supervisedLearningMidterm/
├── .ipynb_checkpoints/   # Checkpoints for Jupyter notebooks
├── data/                 # Contains the dataset files
├── images/               # Images used in the project
├── models/               # Saved models and model-related files
├── notebooks/            # Jupyter notebooks for data analysis and model training
├── .DS_Store             # System file
├── README.md             # Project README file
├── assignment.md         # Assignment details and instructions
```

## Usage

To run the project and evaluate the models, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/imarri01/LHL_supervisedLearningMidterm.git
   cd LHL_supervisedLearningMidterm
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks to train and evaluate the models:

   ```bash
   jupyter notebook
   ```

4. Open the respective notebook in the `notebooks` directory and run the cells to see the results.

## Future Goals

(what would you do if you had more time?)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
