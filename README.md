# Document Similarity Assessment: Quora Question Pair Similarity

## Introduction
This report details the methodology and processes employed to identify duplicate questions on Quora. The objective is to discern whether two questions, although phrased differently, are essentially asking the same thing and therefore likely to elicit identical responses.

## 1. Data Acquisition and Preliminary Analysis
The dataset consists of pairs of questions from Quora. The initial step involves loading the data into our analytical environment, followed by a preliminary examination to understand its structure, identify missing values, and gather key insights that will guide further exploratory and preparatory steps.

## 2. Exploratory Data Analysis (EDA)
The EDA phase aims to provide a deeper understanding of the dataset's characteristics:
- **Distribution Analysis:** Assessing the spread and common patterns in the dataset.
- **Missing Values:** Identifying and addressing gaps in the data.
- **Anomaly Detection:** Highlighting any outliers or inconsistencies that could influence model performance.

This comprehensive analysis ensures that the dataset is well-understood and appropriately prepared for the subsequent preprocessing steps.

## 3. Data Preprocessing
Robust data preprocessing is crucial for the effectiveness of the model:

- **Text Cleaning:** Retention of punctuation marks due to their potential significance in understanding context, coupled with cautious handling of stopwords to avoid loss of critical contextual differences.
- **GloVe Embeddings:** Integration of 300D GloVe embeddings via SpaCy enhances our model’s ability to understand and process semantic similarities and context within the text.
- **Tokenization and Handling Missing Data:** The text is tokenized into discrete units, and missing data points are either imputed or removed based on their impact on the dataset's integrity.

## 4. Feature Engineering
Feature engineering focuses on extracting meaningful attributes from the text data that are indicative of question similarity:

- **Keyword Matching:** Features such as `first_word_match` and `last_word_match` are derived to capture the similarity at the beginning and end of questions.
- **Stopword Analysis:** The `common_stopwords`, `stopword_ratio`, and `common_words` features help in understanding the structural and contextual composition of the questions.
- **Text Length and Character Analysis:** Features like `len_diff`, `total_words`, `common_chars`, and `char_overlap` provide quantitative metrics to assess similarity.

**Correlation Assessment:** A correlation matrix is generated to determine the strength and relevance of each feature with respect to the target variable. Only features with a significant correlation coefficient (above 0.3) are retained for model development.

# 5. Model Preparation and CSV Export
After feature engineering, the dataset is enriched with contextual embeddings and relevant features. This processed dataset is then exported as a CSV file, ready to be uploaded to Google Colab for training the predictive model.

# 6. Model Training

Given the extensive dataset and the need to efficiently manage computational resources and time, our model training adopts a structured, phased approach:

### 1. **Initial Training with 100,000 Samples:**
   - **Objective:** Establish a baseline and identify models with potential high performance.
   - **Models Trained:**
     - **Linear Model:** Provides a straightforward benchmark.
     - **Random Forest:** Explores robustness to overfitting, suitable for initial feature evaluation.
     - **Gradient Boosting and XGBoost:** Efficient with large datasets, used for gaining initial feature importance insights.
     - **Neural Network (Simple Architecture in TensorFlow):** Captures complex nonlinear relationships, showing promising initial results possibly due to its ability to model the nuanced interactions between words effectively at this scale.
     - **Support Vector Machine (SVM):** Effective in high-dimensional data scenarios, important for text classification.

### 2. **Intermediate Training with 200,000 Samples:**
   - **Objective:** Refine models showing promising results and conduct hyperparameter tuning.
   - **Models Trained:**
     - **XGBoost:** Further tuning based on insights from the initial phase.
     - **Optimized Simple Neural Network:** Slight modifications in architecture and parameters based on performance metrics to improve handling of textual data nuances.
     - **Random Forest:** Phased out due to excessive training time.

### 3. **Final Training with Complete Dataset (404,290 Samples):**
   - **Objective:** Utilize the full dataset to maximize accuracy and generalization capability.
   - **Models Trained:**
     - **XGBoost (Optimized):** Enhanced to maximize robustness and efficiency for large-scale applications.
     - **Neural Network (Hypertuned with Keras Tuner):** Despite earlier success, performance declined with increased data volume, indicating potential overfitting as the model complexity may not scale appropriately with data size. Future steps will involve revisiting the network architecture and regularization strategies to prevent overfitting and ensure scalability.

# 7. Model Deployment and Demonstration

After completing the training and evaluation phases, the XGBoost model was selected for deployment:

### Model Saving and Testing:
- **Saving the Model:** The optimized XGBoost model, which consistently performed well across various stages, was saved using the Python `pickle` module. This approach ensures the model can be easily reloaded for future predictions without needing retraining.
- **Model Application:** The saved model was then loaded from its `.pkl` file and applied to the test dataset to evaluate.

### Demonstration of Integration on Quora Platform:
- **Prototype Interface:** To visualize how the model integration would function on the Quora platform, a basic prototype was developed using flask.

# 8. Model Performance Evaluation: XGBoost Classifier

The XGBoost Classifier was chosen as the final model due to its robust performance throughout the training phases. Here is a detailed breakdown of its performance metrics:

### XGBoost Classifier Results:
- **Accuracy:** 0.79
- **F1 Score:** 0.78
- **Log Loss:** 0.43

### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.82      | 0.85   | 0.83     | 50,803  |
| 1     | 0.72      | 0.68   | 0.70     | 30,055  |

### Overall Performance:
- **Accuracy:** 0.79
- **Macro Average:**
  - **Precision:** 0.77
  - **Recall:** 0.76
  - **F1-Score:** 0.77
- **Weighted Average:**
  - **Precision:** 0.78
  - **Recall:** 0.79
  - **F1-Score:** 0.78

### Summary:
The XGBoost classifier achieved an accuracy of 79% and an F1 score of 78%, indicating strong predictive capabilities, especially considering the challenging nature of the task. The log loss of 0.43 suggests the model is well-calibrated, providing reliable probability estimates for its predictions.

The classifier exhibited superior performance in identifying class 0 (non-duplicate questions) with higher precision, recall, and F1-score compared to class 1 (duplicate questions). This discrepancy highlights potential areas for further tuning and adaptation, particularly in improving detection sensitivity for duplicate questions.
