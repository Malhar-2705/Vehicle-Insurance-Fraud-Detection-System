Advanced Car Insurance Fraud Detection
📋 Overview
This project implements a complete machine learning pipeline to predict fraudulent car insurance claims. It begins with a comprehensive Exploratory Data Analysis (EDA) to uncover underlying data patterns and concludes by training a high-performance LightGBM (LGBM) classifier.

A key feature of this project is the use of Bayesian Hyperparameter Optimization via the Optuna framework. This advanced technique intelligently searches for the optimal model parameters, leading to superior performance compared to manual tuning or grid search. The final model is evaluated on multiple metrics to ensure its effectiveness in identifying suspicious claims.

✨ Features
The project script is a single, end-to-end pipeline that performs:

In-Depth EDA:

Univariate Analysis: Histograms, count plots, and a pie chart to understand individual feature distributions and class imbalance.

Bivariate Analysis: Box plots, violin plots, and stacked bar charts to analyze how each feature relates to the fraud outcome.

Multivariate Analysis: A correlation heatmap and a pair plot to visualize complex interactions between numerical features.

Advanced Modeling:

LightGBM Classifier: Utilizes a fast, efficient, and highly accurate gradient boosting model.

Bayesian Hyperparameter Optimization: Employs Optuna to automatically find the best hyperparameters for the LightGBM model, maximizing its predictive power (specifically, the AUC score).

Robust Evaluation:

Classification Metrics: Calculates Accuracy, Precision, Recall, and F1-Score.

Confusion Matrix: Visualizes the model's performance in distinguishing between fraud and non-fraud cases.

Feature Importance Plot: Identifies the key drivers that the model uses to make predictions.

ROC Curve & AUC Score: Provides a comprehensive measure of the model's classification ability across all thresholds.

💾 Dataset
The analysis is based on the carclaims.csv dataset from Kaggle. It contains numerous features related to insurance policies, vehicles, and claimants. The primary target variable for prediction is FraudFound.

🛠️ Technologies Used
Python 3.x

Pandas: For data loading and manipulation.

Matplotlib & Seaborn: For comprehensive data visualization.

Scikit-learn: For data splitting and model evaluation metrics.

LightGBM (LGBM): The core gradient boosting framework for the classification model.

Optuna: For state-of-the-art Bayesian hyperparameter optimization.

🚀 How to Run the Code
Follow these steps to replicate the analysis on your local machine.

Clone the Repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install Dependencies:
It is highly recommended to use a virtual environment.

pip install pandas matplotlib seaborn scikit-learn lightgbm optuna

Place the Dataset:
Ensure the carclaims.csv file is in the same directory as the script, or update the file_path variable in the code to its correct location.

Execute the Script:
Run the script from your terminal. The process will take several minutes due to the hyperparameter optimization search.

python your_script_name.py

The script will first display all the EDA plots. After the plots are closed, it will begin the Optuna optimization process, printing the progress of each trial. Finally, it will output the model's performance metrics and display the evaluation plots.

📊 Results and Key Findings
The LightGBM model, tuned with Bayesian optimization, provides a robust solution for fraud detection. The final classification report details the model's precision and recall, which are critical metrics for imbalanced datasets like this one.

The key outputs to analyze are:

Best Hyperparameters: The script prints the optimal set of parameters found by Optuna, which can be reused for future model training.

Feature Importance Plot: This visualization reveals the most significant predictors of fraud. Factors like PolicyType, VehicleCategory, and DriverRating are often highly influential.

Confusion Matrix: This shows the exact number of fraudulent claims that were correctly identified (True Positives) and those that were missed (False Negatives), which is crucial for assessing real-world impact.

🏁 Conclusion
This project successfully builds and optimizes an advanced classification model for a real-world problem. It demonstrates the power of combining a strong algorithm like LightGBM with an intelligent tuning strategy like Bayesian optimization.

Potential next steps could include:

Implementing techniques to handle class imbalance more explicitly (e.g., SMOTE).

Deploying the trained model as a REST API for real-time predictions.

Integrating other advanced models like CatBoost or XGBoost into the optimization pipeline for comparison.
