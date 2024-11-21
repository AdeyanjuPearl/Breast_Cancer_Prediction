# Breast_Cancer_Prediction

### Project Overview-
The project aims to build a KNN model to classify a patient's breast mass as either benign or malignant based on the clinical features recorded after analyzing the cell nuclei of the breast mass through fine needle aspiration.

### Data Source:
The Wiscosin Breast Cancer dataset downloaded from Kaggle was used for this project. The dataset consist of 11 columns and 699 rows.

### Tools: 
- Excel: for data warehousing
- Python: for Model development, training and evaluation
-Jupyter Notebook

### Data Exploration

After creating a Jupyter Notebook file for this project, the following data exporation tasks were performed:
Data loading and inspection through statistical summary and graphical view.
Identifying Handling outliers
Checking for missing values
Removing redundant variables
Converting data types

### Data Visualisation

From the Data preprocessing step, I was able to garnered the basic understanding of the dataset. However, I will use visuals to get a deeper insight by plotting the following graphs
-Univariate-Histogram
-Multivariate- Correlation heat map

### Data Preprocessing
-Declaring the Feature Vector and the Target Class
-Splitting the dataset into training and test set
-Feature Engineering- Handling missing values using median imputation assuming they are missing completely at random.
-Feature Scaling

### Model Development
- Model training: the feature vectors and the Target Class in the training set are fed into the KNeigbour Classifier using K=3
- The model was used to Predict the test set result.
- The predict_proba method was used to obtain the probability of getting the target variable

### Model Evaluation
- Checking accuracy score
- Checking for overfitting and underfitting
- Rebuilding KNN Classification Model using different values of K and checking for the accuracy score of each K Value
- Constructing the Confusion Matrix
- finding the classification report

### [View my code](Breast Cancer Prediction.ipynb)
 
