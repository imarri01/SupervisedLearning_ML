


![banner](/images/LHL_banner.png)

---

# LHL Supervised Learning Midterm

## 🌟 Overview

The goal of this project is to explore and implement different supervised learning regression models to predict the price of houses based on various features such as square footage, number of bedrooms, number of bathrooms, and other relevant attributes. We evaluated the performance of these models using various metrics to determine the best model for this task.

## 👥 Group Members

- **Ramon Kidd:** Contributor
- **Kiran Reid:** Contributor


## 🛠️ Project Steps

### 🧹 Data Cleaning

- **Source of Data:** The dataset includes various features relevant to predicting house prices.
- **Raw Data:** Located in the `data/raw_json_files` directory.
- **Processed Data:** Located in the `data/processed` directory with the main file being `raw_merged_df_data.csv`.
- **Actions:**
  - Loaded the raw data from JSON files.
  - Merged multiple data sources into a single dataframe.
  - Handled missing values by appropriate imputation methods.
  - Converted categorical variables into numerical format using encoding techniques.

### 📊 Exploratory Data Analysis (EDA)

- **Notebooks:** EDA steps are documented in `notebooks/1 - EDA.ipynb`.
- **Visualizations:**
  - Various bar graphs (`images/bar_graphs_categorical.png`, `images/bar_graphs_numerical.png`).
  - Box plots (`images/box_plots.png`).
  - Correlation heatmaps (`images/correlation_heatmap_new.png`, `images/correlation_heatmap.png`).
  - Pair plot (`images/pair_plot.png`).
  - Price distribution plot (`images/price_distribution.png`).
- **Actions:**
  - Analyzed the distribution of key features.
  - Examined relationships between features and the target variable (house prices).
  - Identified potential outliers and anomalies.

### 🔍 Model Selection

- **Notebooks:** Model selection steps are documented in `notebooks/2 - model-selection.ipynb`.
- **Models Tested:**
  - Decision Tree
  - Random Forest
  - Other relevant regression models
- **Actions:**
  - Split the data into training and testing sets.
  - Trained multiple models on the training set.
  - Evaluated models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
  - Selected the best-performing models for further tuning.

### 🔧 Model Tuning

- **Notebooks:** Tuning steps are documented in `notebooks/3 - tunning_pipeline.ipynb`.
- **Actions:**
  - Hyperparameter tuning using techniques such as Grid Search and Random Search.
  - Validated model performance using cross-validation.
  - Finalized the best model based on performance metrics.

### 🚀 MLOps Pipeline

- **Pipeline:** Located in the `models/pipeline` directory.
- **Files:**
  - `decision_tree_pipeline.pkl`
  - `final_pipeline.pkl`
- **Actions:**
  - Created a scalable pipeline for model training and deployment.
  - Automated the data preprocessing, model training, and evaluation process.
  - Saved the final model and pipeline for deployment.

## 📈 Results

Based on the performance metrics selected, the **Decision Tree Regressor** performed the best across all the models. See details results and comparison below, along with summary and next steps for Part 3 of the project:

##### ***Best Performing Model: Decision Tree Regressor***

- **Mean Squared Error (MSE)**: 6,193,112,904.08
- **Mean Absolute Error (MAE)**: 8,933.19
- **R² Score**: 0.9864

##### ***Comparison with Other Models***

| Model                         | Mean Squared Error (MSE)  | Mean Absolute Error (MAE)  | R² Score  |
|-------------------------------|---------------------------|----------------------------|-----------|
| **Decision Tree**             | 6,193,112,904.08          | 8,933.19                   | 0.9864    |
| **Random Forest**             | 6,612,175,921.27          | 24,107.06                  | 0.9855    |
| **XGBoost**                   | 4,439,562,534.91          | 21,193.02                  | 0.9902    |
| **Linear Regression**         | 274,501,793,270.35        | 204,038.20                 | 0.3967    |
| **Support Vector Machine**    | 483,900,366,034.21        | 250,867.79                 | -0.0636   |
| **K-Nearest Neighbors**       | 77,604,134,366.93         | 97,416.02                  | 0.8294    |
| **Gradient Boosting**         | 34,976,229,429.47         | 109,245.70                 | 0.9231    |
| **AdaBoost**                  | 87,353,813,837.12         | 247,798.41                 | 0.8080    |
| **Ridge Regression**          | 274,532,623,656.19        | 203,986.27                 | 0.3966    |
| **Lasso Regression**          | 274,502,594,826.23        | 204,035.61                 | 0.3967    |
| **ElasticNet Regression**     | 291,398,512,534.51        | 198,455.73                 | 0.3595    |


## 🗂️ Directory Structure

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

## 🚀 Usage

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


---


