


![banner](/images/LHL_banner.png)

---

# LHL Supervised Learning Midterm

## 🌟 Overview

The goal of this project is to explore and implement different supervised learning regression models to predict the price of houses based on various features such as square footage, number of bedrooms, number of bathrooms, and other relevant attributes. We evaluated the performance of these models using various metrics to determine the best model and parameters for this task.

## 👥 Group Members

- **Ramon Kidd:** Contributor
- **Kiran Wood:** Contributor


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
  - Split the data into training and test sets

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

- **Notebooks:** Model selection steps are documented in `notebooks/2 - model_selection.ipynb`.
- **Models Tested:**
  - Decision Tree
  - Random Forest
  - Other relevant regression models
- **Actions:**
  - Split the training data into training and validation sets.
  - Trained multiple models on the training set.
  - Evaluated models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
  - Selected the best-performing models for further tuning.

### 🔧 Model Tuning

- **Notebooks:** Tuning steps are documented in `notebooks/3 - tunning_pipeline.ipynb`.
- **Actions:**
  - Hyperparameter tuning using techniques such as Grid Search and Random Search.
  - Validated model performance using cross-validation.
  - Finalized the best model based on performance metrics.
  - Compared different tuned models to see which preformed better
  - Selected best model for feature selection

### 🪚 Feature Selection

- **Notebooks:** Tuning steps are documented in `notebooks/3 - tunning_pipeline.ipynb`.
- **Actions:**
  - Analyzed the top 10 important features of best model
  - Used visualization (`images/top_features_graph.png`)
  - Compared a model with only most important features vs one without
  - Chose best model to move onto final pipeline

### 🚀 MLOps Pipeline

- **Pipeline:** Located in the `models/pipeline` directory.
- **Files:**
  - `final_pipeline.pkl`
- **Actions:**
  - Created a scalable pipeline for model training and deployment.
  - Automated the data preprocessing, model training, and evaluation process.
  - Saved the final model/pipeline for deployment.

## 📈 Results

After all the comparisons and finetuning, the **Decision Tree Regressor** was the best performeing across all the models. These are the parameters used and results on test data:

##### ***Best Performing Model: Decision Tree Regressor***

- **Mean Squared Error (MSE)**: 3,366,377,871.16
- **Mean Absolute Error (MAE)**: 10,258.26
- **R² Score**: 0.9898

##### ***Model and Parameter Details***

| Best Model                         | Parameters  | Preprocessing  |
|-------------------------------|---------------------------|----------------------------|
| **Decision Tree**             | `max_depth`: None, `min_samples_leaf`: 1, `min_samples_split`: 2          | `Numerical Data`: StandardScaler, `Categorical Data`: OneHotEncoding     |


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
├── old_files             # Old files made during project progress
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

## 💬 Challenges

- Separating the `tag` category into different columns
- Hyperparameter tuning the models
- Deciding how to encode the `city` category

## ✨ Future Goals

- Explore different preprocessing methods
- Search for data that could potentially be added to the model in order to achieve better results
- Try more unique models to see how they will work with the data

---


