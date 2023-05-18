
## Overview of the Analysis

* The financial information used was lending data, presumably collected from lender databases, and we need to model loan status as labels versus every other column which are used as features to create the pattern to predict from.
* The variables predicted use a pattern generated from 75036 healthy loans versus 2500 high risk loans.
* The stages of the machine learning process traversed as part of this analysis were:
  1. Separating the data as features vs labels
  2. Splitting the data into training and testing datasets by using `train_test_split` from scikit learn
  3. Instantiating, fitting, and saving a model based on training data
  4. Making predictions on the fitted model using testing data
  5. Generating scores, reports, graphs etc. based on the saved predictions
* LogisticRegression was the primary method used for modeling. The original data was used in the 1st instance and the RandomOverSampler modified the data before being used in combination with the aforementioned model in the 2nd instance.

* Machine Learning Logistic Regression Model with the Original Data:
  * The Accuracy score using the balaced_accuracy_score in scikit learn resulted in 95.2% accuracy. 
  * Precision was 100% for (0)healthy loans and 85% for (1)high risk loan labels
  * Recall was 99% for (0) and 91% for (1)


* Machine Learning Logistic Regression Model with Resampled Training Data:
  * The Accuracy score using the balaced_accuracy_score in scikit learn resulted in 99.3% accuracy 
  * Precision was 100% for (0)healthy loans and 84% for (1)high risk loan labels 
  * Recall was 99% for both (0) and (1)
 ## Summary

* The Logistic Regression Model with Resampled Training Data seems to perform best, because it predicts high risk loan labels with a greater recall and F1 scores
* However, performance does depend on the problem we are trying to solve, because there is negligable gain from using the resampled training data to predict healthy loans seeing as the recall and F1 scores are almost identical for (0).
Therefore, I would recomment the Logistic Regression Model with Resampled Training Data for predicting high risk loans only.

To Run: Python with the libraries, pandas, scikit learn, and imblearn(for RandomOversampling) are required. Jupyter Notebook is recommended to view and run cells.
