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

### [View my code](breast-cancer-predictio{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "20d0c091-3aed-40b9-bcd3-71536af08ce7",
   "metadata": {},
   "source": [
    "KNN "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "11a95c23-e47e-4e7e-9ed7-2b9c6f67fa7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Importing all libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1f4759d3-6a2a-4797-a4c5-c9497fd83c5c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\MACBOOK AIR'"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###Confirming the Current Working Directory\n",
    "import os\n",
    "cwd= os.getcwd()\n",
    "cwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "36808512-0522-44f6-853e-e2b7ce28321e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>sepal_length</td>\n",
       "      <td>sepal_width</td>\n",
       "      <td>petal_length</td>\n",
       "      <td>petal_width</td>\n",
       "      <td>species</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              0            1             2            3            4\n",
       "0  sepal_length  sepal_width  petal_length  petal_width      species\n",
       "1           5.1          3.5           1.4          0.2  Iris-setosa\n",
       "2           4.9            3           1.4          0.2  Iris-setosa\n",
       "3           4.7          3.2           1.3          0.2  Iris-setosa\n",
       "4           4.6          3.1           1.5          0.2  Iris-setosa"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###Importing Iris Dataset\n",
    "Data = \"C:/Users\\MACBOOK AIR/Desktop/IRIS.csv\"\n",
    "df= pd.read_csv(Data, header= None)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "813203ae-40d1-43a5-845a-e4491b6c87c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Sample code number</td>\n",
       "      <td>Clump Thickness</td>\n",
       "      <td>Uniformity of Cell Size</td>\n",
       "      <td>Uniformity of Cell Shape</td>\n",
       "      <td>Marginal Adhesion</td>\n",
       "      <td>Single Epithelial Cell Size</td>\n",
       "      <td>Bare Nuclei</td>\n",
       "      <td>Bland Chromatin</td>\n",
       "      <td>Normal Nucleoli</td>\n",
       "      <td>Mitoses</td>\n",
       "      <td>Class</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000025</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1002945</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1015425</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1016277</td>\n",
       "      <td>6</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   0                1                        2   \\\n",
       "0  Sample code number  Clump Thickness  Uniformity of Cell Size   \n",
       "1             1000025                5                        1   \n",
       "2             1002945                5                        4   \n",
       "3             1015425                3                        1   \n",
       "4             1016277                6                        8   \n",
       "\n",
       "                         3                  4                            5   \\\n",
       "0  Uniformity of Cell Shape  Marginal Adhesion  Single Epithelial Cell Size   \n",
       "1                         1                  1                            2   \n",
       "2                         4                  5                            7   \n",
       "3                         1                  1                            2   \n",
       "4                         8                  1                            3   \n",
       "\n",
       "            6                7                8        9      10  \n",
       "0  Bare Nuclei  Bland Chromatin  Normal Nucleoli  Mitoses  Class  \n",
       "1            1                3                1        1      2  \n",
       "2           10                3                2        1      2  \n",
       "3            2                3                1        1      2  \n",
       "4            4                3                7        1      2  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###Importing Breast Cancer Wisconsin Dataset\n",
    "Data = \"C:/Users/MACBOOK AIR/Desktop/breast_cancer_bd.csv\"\n",
    "df2= pd.read_csv(Data, header= None)\n",
    "df2.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e240184c-92e6-4817-af58-1ef80d9f0002",
   "metadata": {},
   "source": [
    "Performing Exploratory Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "28446560-e346-42a8-8d3a-1b519c1ea54f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 700 entries, 0 to 699\n",
      "Data columns (total 11 columns):\n",
      " #   Column  Non-Null Count  Dtype \n",
      "---  ------  --------------  ----- \n",
      " 0   0       700 non-null    object\n",
      " 1   1       700 non-null    object\n",
      " 2   2       700 non-null    object\n",
      " 3   3       700 non-null    object\n",
      " 4   4       700 non-null    object\n",
      " 5   5       700 non-null    object\n",
      " 6   6       700 non-null    object\n",
      " 7   7       700 non-null    object\n",
      " 8   8       700 non-null    object\n",
      " 9   9       700 non-null    object\n",
      " 10  10      700 non-null    object\n",
      "dtypes: object(11)\n",
      "memory usage: 60.3+ KB\n"
     ]
    }
   ],
   "source": [
    "##df2.columns\n",
    "df2.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e471225-22f7-497e-b440-3ad3791acd56",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e4479df5-7a73-4159-a11f-f1bc826d55d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Clump Thickness</th>\n",
       "      <th>Uniformity of Cell Size</th>\n",
       "      <th>Uniformity of Cell Shape</th>\n",
       "      <th>Marginal Adhension</th>\n",
       "      <th>Single Epithelial Cell Size</th>\n",
       "      <th>Bare Nuclei</th>\n",
       "      <th>Bland Chromatin</th>\n",
       "      <th>Normal Nucleoli</th>\n",
       "      <th>Mitoses</th>\n",
       "      <th>Clas</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Sample code number</td>\n",
       "      <td>Clump Thickness</td>\n",
       "      <td>Uniformity of Cell Size</td>\n",
       "      <td>Uniformity of Cell Shape</td>\n",
       "      <td>Marginal Adhesion</td>\n",
       "      <td>Single Epithelial Cell Size</td>\n",
       "      <td>Bare Nuclei</td>\n",
       "      <td>Bland Chromatin</td>\n",
       "      <td>Normal Nucleoli</td>\n",
       "      <td>Mitoses</td>\n",
       "      <td>Class</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000025</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1002945</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1015425</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1016277</td>\n",
       "      <td>6</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   ID  Clump Thickness  Uniformity of Cell Size  \\\n",
       "0  Sample code number  Clump Thickness  Uniformity of Cell Size   \n",
       "1             1000025                5                        1   \n",
       "2             1002945                5                        4   \n",
       "3             1015425                3                        1   \n",
       "4             1016277                6                        8   \n",
       "\n",
       "   Uniformity of Cell Shape Marginal Adhension  Single Epithelial Cell Size  \\\n",
       "0  Uniformity of Cell Shape  Marginal Adhesion  Single Epithelial Cell Size   \n",
       "1                         1                  1                            2   \n",
       "2                         4                  5                            7   \n",
       "3                         1                  1                            2   \n",
       "4                         8                  1                            3   \n",
       "\n",
       "   Bare Nuclei  Bland Chromatin  Normal Nucleoli  Mitoses   Clas  \n",
       "0  Bare Nuclei  Bland Chromatin  Normal Nucleoli  Mitoses  Class  \n",
       "1            1                3                1        1      2  \n",
       "2           10                3                2        1      2  \n",
       "3            2                3                1        1      2  \n",
       "4            4                3                7        1      2  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "col_names= [\"ID\", \"Clump Thickness\", \"Uniformity of Cell Size\", \"Uniformity of Cell Shape\", \"Marginal Adhension\", \"Single Epithelial Cell Size\", \"Bare Nuclei\", \"Bland Chromatin\", \"Normal Nucleoli\", \"Mitoses\", \"Clas\"]\n",
    "df2.columns=col_names\n",
    "df2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cd2970b8-ed3f-44b8-bf28-274153e975d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Clump Thickness</th>\n",
       "      <th>Uniformity of Cell Size</th>\n",
       "      <th>Uniformity of Cell Shape</th>\n",
       "      <th>Marginal Adhension</th>\n",
       "      <th>Single Epithelial Cell Size</th>\n",
       "      <th>Bare Nuclei</th>\n",
       "      <th>Bland Chromatin</th>\n",
       "      <th>Normal Nucleoli</th>\n",
       "      <th>Mitoses</th>\n",
       "      <th>Clas</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000025</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1002945</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1015425</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1016277</td>\n",
       "      <td>6</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1017023</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        ID Clump Thickness Uniformity of Cell Size Uniformity of Cell Shape  \\\n",
       "1  1000025               5                       1                        1   \n",
       "2  1002945               5                       4                        4   \n",
       "3  1015425               3                       1                        1   \n",
       "4  1016277               6                       8                        8   \n",
       "5  1017023               4                       1                        1   \n",
       "\n",
       "  Marginal Adhension Single Epithelial Cell Size Bare Nuclei Bland Chromatin  \\\n",
       "1                  1                           2           1               3   \n",
       "2                  5                           7          10               3   \n",
       "3                  1                           2           2               3   \n",
       "4                  1                           3           4               3   \n",
       "5                  3                           2           1               3   \n",
       "\n",
       "  Normal Nucleoli Mitoses Clas  \n",
       "1               1       1    2  \n",
       "2               2       1    2  \n",
       "3               1       1    2  \n",
       "4               7       1    2  \n",
       "5               1       1    2  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Dropping Redundant tables that has no predictive power on the outcome variable\n",
    "newdf=df2.drop(0, axis=0)\n",
    "newdf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e346eb37-d206-4e9e-8cb4-7ad9266db046",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(699, 11)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newdf.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "25aabdc3-87ec-4b9a-986c-f50f99a15efa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Clump Thickness</th>\n",
       "      <th>Uniformity of Cell Size</th>\n",
       "      <th>Uniformity of Cell Shape</th>\n",
       "      <th>Marginal Adhension</th>\n",
       "      <th>Single Epithelial Cell Size</th>\n",
       "      <th>Bare Nuclei</th>\n",
       "      <th>Bland Chromatin</th>\n",
       "      <th>Normal Nucleoli</th>\n",
       "      <th>Mitoses</th>\n",
       "      <th>Clas</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Clump Thickness Uniformity of Cell Size Uniformity of Cell Shape  \\\n",
       "1               5                       1                        1   \n",
       "2               5                       4                        4   \n",
       "3               3                       1                        1   \n",
       "4               6                       8                        8   \n",
       "5               4                       1                        1   \n",
       "\n",
       "  Marginal Adhension Single Epithelial Cell Size Bare Nuclei Bland Chromatin  \\\n",
       "1                  1                           2           1               3   \n",
       "2                  5                           7          10               3   \n",
       "3                  1                           2           2               3   \n",
       "4                  1                           3           4               3   \n",
       "5                  3                           2           1               3   \n",
       "\n",
       "  Normal Nucleoli Mitoses Clas  \n",
       "1               1       1    2  \n",
       "2               2       1    2  \n",
       "3               1       1    2  \n",
       "4               7       1    2  \n",
       "5               1       1    2  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newdf.drop(\"ID\", axis=1, inplace=True)\n",
    "newdf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7a29391e-6322-460f-af07-e915ba2eb6fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 699 entries, 1 to 699\n",
      "Data columns (total 10 columns):\n",
      " #   Column                       Non-Null Count  Dtype \n",
      "---  ------                       --------------  ----- \n",
      " 0   Clump Thickness              699 non-null    object\n",
      " 1   Uniformity of Cell Size      699 non-null    object\n",
      " 2   Uniformity of Cell Shape     699 non-null    object\n",
      " 3   Marginal Adhension           699 non-null    object\n",
      " 4   Single Epithelial Cell Size  699 non-null    object\n",
      " 5   Bare Nuclei                  699 non-null    object\n",
      " 6   Bland Chromatin              699 non-null    object\n",
      " 7   Normal Nucleoli              699 non-null    object\n",
      " 8   Mitoses                      699 non-null    object\n",
      " 9   Clas                         699 non-null    object\n",
      "dtypes: object(10)\n",
      "memory usage: 54.7+ KB\n"
     ]
    }
   ],
   "source": [
    "#Viewing the summary of the dataset\n",
    "newdf.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "12fd446d-2c7b-4c32-9433-d42fe57fed5b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Clump Thickness\n",
      "1     145\n",
      "5     130\n",
      "3     108\n",
      "4      80\n",
      "10     69\n",
      "2      50\n",
      "8      46\n",
      "6      34\n",
      "7      23\n",
      "9      14\n",
      "Name: count, dtype: int64\n",
      "Uniformity of Cell Size\n",
      "1     384\n",
      "10     67\n",
      "3      52\n",
      "2      45\n",
      "4      40\n",
      "5      30\n",
      "8      29\n",
      "6      27\n",
      "7      19\n",
      "9       6\n",
      "Name: count, dtype: int64\n",
      "Uniformity of Cell Shape\n",
      "1     353\n",
      "2      59\n",
      "10     58\n",
      "3      56\n",
      "4      44\n",
      "5      34\n",
      "6      30\n",
      "7      30\n",
      "8      28\n",
      "9       7\n",
      "Name: count, dtype: int64\n",
      "Marginal Adhension\n",
      "1     407\n",
      "3      58\n",
      "2      58\n",
      "10     55\n",
      "4      33\n",
      "8      25\n",
      "5      23\n",
      "6      22\n",
      "7      13\n",
      "9       5\n",
      "Name: count, dtype: int64\n",
      "Single Epithelial Cell Size\n",
      "2     386\n",
      "3      72\n",
      "4      48\n",
      "1      47\n",
      "6      41\n",
      "5      39\n",
      "10     31\n",
      "8      21\n",
      "7      12\n",
      "9       2\n",
      "Name: count, dtype: int64\n",
      "Bare Nuclei\n",
      "1     402\n",
      "10    132\n",
      "2      30\n",
      "5      30\n",
      "3      28\n",
      "8      21\n",
      "4      19\n",
      "?      16\n",
      "9       9\n",
      "7       8\n",
      "6       4\n",
      "Name: count, dtype: int64\n",
      "Bland Chromatin\n",
      "2     166\n",
      "3     165\n",
      "1     152\n",
      "7      73\n",
      "4      40\n",
      "5      34\n",
      "8      28\n",
      "10     20\n",
      "9      11\n",
      "6      10\n",
      "Name: count, dtype: int64\n",
      "Normal Nucleoli\n",
      "1     443\n",
      "10     61\n",
      "3      44\n",
      "2      36\n",
      "8      24\n",
      "6      22\n",
      "5      19\n",
      "4      18\n",
      "7      16\n",
      "9      16\n",
      "Name: count, dtype: int64\n",
      "Mitoses\n",
      "1     579\n",
      "2      35\n",
      "3      33\n",
      "10     14\n",
      "4      12\n",
      "7       9\n",
      "8       8\n",
      "5       6\n",
      "6       3\n",
      "Name: count, dtype: int64\n",
      "Clas\n",
      "2    458\n",
      "4    241\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "for var in newdf.columns:\n",
    "    print(newdf[var].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5309bf4c-3bef-4b0a-9ba0-db44109a9eee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 699 entries, 1 to 699\n",
      "Data columns (total 10 columns):\n",
      " #   Column                       Non-Null Count  Dtype \n",
      "---  ------                       --------------  ----- \n",
      " 0   Clump Thickness              699 non-null    int64 \n",
      " 1   Uniformity of Cell Size      699 non-null    int64 \n",
      " 2   Uniformity of Cell Shape     699 non-null    int64 \n",
      " 3   Marginal Adhension           699 non-null    int64 \n",
      " 4   Single Epithelial Cell Size  699 non-null    int64 \n",
      " 5   Bare Nuclei                  699 non-null    object\n",
      " 6   Bland Chromatin              699 non-null    object\n",
      " 7   Normal Nucleoli              699 non-null    object\n",
      " 8   Mitoses                      699 non-null    object\n",
      " 9   Clas                         699 non-null    object\n",
      "dtypes: int64(5), object(5)\n",
      "memory usage: 54.7+ KB\n"
     ]
    }
   ],
   "source": [
    " ##Converting data type of all columns from object to numeric\n",
    "newdf[[\"Clump Thickness\", \"Uniformity of Cell Size\", \"Uniformity of Cell Shape\", \"Marginal Adhension\", \"Single Epithelial Cell Size\"]]= newdf[[\"Clump Thickness\", \"Uniformity of Cell Size\", \"Uniformity of Cell Shape\", \"Marginal Adhension\", \"Single Epithelial Cell Size\"]].apply(pd.to_numeric)\n",
    "newdf.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6cc3c8fe-0b73-464d-a719-8c9b3a614957",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Converting the remaining columns to numeric\n",
    "newdf[[\"Bare Nuclei\", \"Bland Chromatin\", \"Normal Nucleoli\", \"Mitoses\", \"Clas\"]]=newdf[[\"Bare Nuclei\", \"Bland Chromatin\", \"Normal Nucleoli\", \"Mitoses\", \"Clas\"]].apply(pd.to_numeric, errors='coerce')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "757d4e03-69f4-4a01-8e5b-7439ca8d67e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 699 entries, 1 to 699\n",
      "Data columns (total 10 columns):\n",
      " #   Column                       Non-Null Count  Dtype  \n",
      "---  ------                       --------------  -----  \n",
      " 0   Clump Thickness              699 non-null    int64  \n",
      " 1   Uniformity of Cell Size      699 non-null    int64  \n",
      " 2   Uniformity of Cell Shape     699 non-null    int64  \n",
      " 3   Marginal Adhension           699 non-null    int64  \n",
      " 4   Single Epithelial Cell Size  699 non-null    int64  \n",
      " 5   Bare Nuclei                  683 non-null    float64\n",
      " 6   Bland Chromatin              699 non-null    int64  \n",
      " 7   Normal Nucleoli              699 non-null    int64  \n",
      " 8   Mitoses                      699 non-null    int64  \n",
      " 9   Clas                         699 non-null    int64  \n",
      "dtypes: float64(1), int64(9)\n",
      "memory usage: 54.7 KB\n"
     ]
    }
   ],
   "source": [
    "newdf.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4d62a98f-f816-4ded-af27-87986d778654",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clump Thickness                 0\n",
       "Uniformity of Cell Size         0\n",
       "Uniformity of Cell Shape        0\n",
       "Marginal Adhension              0\n",
       "Single Epithelial Cell Size     0\n",
       "Bare Nuclei                    16\n",
       "Bland Chromatin                 0\n",
       "Normal Nucleoli                 0\n",
       "Mitoses                         0\n",
       "Clas                            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###Checking for missing values\n",
    "newdf.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7ffcf6dd-366e-4419-ab14-2e7442bf8624",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Bare Nuclei\n",
       "1.0     402\n",
       "10.0    132\n",
       "2.0      30\n",
       "5.0      30\n",
       "3.0      28\n",
       "8.0      21\n",
       "4.0      19\n",
       "9.0       9\n",
       "7.0       8\n",
       "6.0       4\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the frequency distribution of Bare Nuclei column\n",
    "newdf[\"Bare Nuclei\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a409b11f-7852-4226-9237-87628264fcef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1., 10.,  2.,  4.,  3.,  9.,  7., nan,  5.,  8.,  6.])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking for unique values in the Bare Nuclei column\n",
    "newdf[\"Bare Nuclei\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9f12ce85-7a30-4bfd-9ee3-dc28e615cec6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clas\n",
       "2    458\n",
       "4    241\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Checking the frequency distribution of the Class column\n",
    "newdf[\"Clas\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0ab128c7-09a7-4779-b43d-7996d2e68682",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clas\n",
       "2    0.655222\n",
       "4    0.344778\n",
       "Name: count, dtype: float64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newdf[\"Clas\"].value_counts()/(float(len(newdf)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d12deb26-3e4d-4899-b042-2d066aea8a81",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       Clump Thickness  Uniformity of Cell Size  Uniformity of Cell Shape  \\\n",
      "count           699.00                   699.00                    699.00   \n",
      "mean              4.42                     3.13                      3.21   \n",
      "std               2.82                     3.05                      2.97   \n",
      "min               1.00                     1.00                      1.00   \n",
      "25%               2.00                     1.00                      1.00   \n",
      "50%               4.00                     1.00                      1.00   \n",
      "75%               6.00                     5.00                      5.00   \n",
      "max              10.00                    10.00                     10.00   \n",
      "\n",
      "       Marginal Adhension  Single Epithelial Cell Size  Bare Nuclei  \\\n",
      "count              699.00                       699.00       683.00   \n",
      "mean                 2.81                         3.22         3.54   \n",
      "std                  2.86                         2.21         3.64   \n",
      "min                  1.00                         1.00         1.00   \n",
      "25%                  1.00                         2.00         1.00   \n",
      "50%                  1.00                         2.00         1.00   \n",
      "75%                  4.00                         4.00         6.00   \n",
      "max                 10.00                        10.00        10.00   \n",
      "\n",
      "       Bland Chromatin  Normal Nucleoli  Mitoses    Clas  \n",
      "count           699.00           699.00   699.00  699.00  \n",
      "mean              3.44             2.87     1.59    2.69  \n",
      "std               2.44             3.05     1.72    0.95  \n",
      "min               1.00             1.00     1.00    2.00  \n",
      "25%               2.00             1.00     1.00    2.00  \n",
      "50%               3.00             1.00     1.00    2.00  \n",
      "75%               5.00             4.00     1.00    4.00  \n",
      "max              10.00            10.00    10.00    4.00  \n"
     ]
    }
   ],
   "source": [
    "##Checking the summary statistics of the numerical variables to check for outliers\n",
    "print(round(newdf.describe(), 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c01d2ded-2fbb-4ead-b524-59243b1ef4d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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+OKLmTdvXgYMGJCvfvWrefHFF3Pbbbela9eu+dGPfpR+/frlq1/9ah566KH06tUrkyZNysEHH9zsWm+88UYmT56cyy+/vFhPkkyaNCmPPfZYFi9enF/96lfFe33nnXey++67Z9SoURvsH/7yl7/Md7/73bz00kvp3r17LrzwwgwfPrz4c1jfvf+tcePG5bHHHsuTTz6ZAw88sDj+4Q9/OJ/73OeyatWqJEmhUMjVV1+dn/zkJ1m0aFH69OmT73znO/nHf/zHkv4up02blu7du2fMmDHFsb//+7/P0UcfXdz/659dsuFV+t9bIe21117LsGHD8sADD6RVq1Y5/PDD88Mf/jA9e/YsqcaNsemRPAAAAKAkf/nLX3Lffffl/PPPbxa+es/6Qkyb4q233sqPfvSjTJ06Nffdd18effTRnHzyybnnnntyzz335Kabbsr111+f//zP/2x23tVXX5399tsvTz31VEaOHJlvfvObmTZt2mbVkiSXX355Tj311DzzzDM55phj8sUvfjF/+ctf1jt3zpw5GThwYPbZZ5/893//d2bMmJHjjz++uNrWX5s6dWpOPfXU3HjjjTnzzDOL44888khefvnlPPLII5kyZUomT56cyZMnF49/+ctfzuOPP56pU6fmmWeeyec+97kcffTRmTdvXpLk/PPPT1NTUx577LE8++yzueqqq7LzzjsnSb7zne/kd7/7Xe699948//zzmThxYnbbbbfN/hkBAAAAUJqXX345d9xxR371q1/lV7/6VaZPn54f/OAH651722235Wtf+1oOPfTQLFq0KLfddluS5Oyzz86TTz6Zu+66K//93/+dQqGQY445JqtXry6e+9Zbb2X06NH5t3/7t8ydOzfdunVLklx77bU57LDD8tvf/jbHHntszjjjjJx55pn50pe+lKeeeiof+chHcuaZZ673tYmnnXZahg8fnn322SeLFi3KokWLctppp+WrX/1q7rvvvixatKg495577smKFSty6qmnrvfeZs+enVNPPTWf//zn8+yzz6ahoSHf+c53in2xDd373/r5z3+eQYMGNQtfvadNmzbFfuY///M/Z9KkSZk4cWLmzp2bb37zm/nSl75UDHltqrq6uixatCiPPfbYRp8za9as4s/t1Vdfzac+9an8wz/8Q5J3/74GDBiQnXfeOY899lhmzJiRnXfeOUcffXQxRNYSrIAFAAAAW8hLL72UQqGQj370oy1y/dWrV2fixIn5+7//+yTJP/7jP+amm27KH//4x+y8887Ze++9M2DAgDzyyCM57bTTiucddthhufTSS5Mkffr0yeOPP55rr702Rx555GbVc/bZZ+cLX/hCkmTUqFEZP358fvOb3zR7eu09Y8aMycEHH5zrrruuOLbPPvusM++6667LZZddljvvvDMDBgxodmzXXXfNhAkT0rp163z0ox/Nsccem4ceeihf+9rX8vLLL+fWW2/Nq6++mvr6+iTJiBEjct9992XSpEkZNWpUXnnllZxyyinZd999k7z7dN97XnnllRx44IHFJxZb8mk5AAAAAD7Y2rVrM3ny5OKKV2eccUYeeuihXHnllevM7dKlS9q3b5+2bdumrq4uSTJv3rzcddddefzxx9OvX78k74aQevTokTvuuCOf+9znkrzbc7vuuuuy//77N7vmMccck3PPPTdJ8t3vfjcTJ07MJz7xieJ5l1xySQ499ND88Y9/LH7me9q1a5edd9451dXVzY7169cve+21V2666aZcfPHFSd5dGetzn/tc8UHBvzV27NgMHDgw3/nOd5K829/73e9+l6uvvjpnn332eu99febNm1dcKX5D3nzzzYwdOzYPP/xwDj300CTv9tBmzJiRn/70pzniiCPe9/z1+dznPpf7778/RxxxROrq6vKpT30qAwcOzJlnnplOnTqt95y/XqX/oosuyqJFizJr1qwk7z682apVq/zbv/1bs9XFdtlllzz66KMZPHjwJte4MayABQAAAFvIe0+7vfcP/3Jr3759MXyVJLW1tenZs2ez5kxtbW0WL17c7Lz3miV/vf/8889vdj377bdf8c8dOnRIx44d1/ns97y3Atb7+eUvf5mhQ4fmgQceWCd8lbwb2Hrv1YpJ0r179+LnPfXUUykUCunTp0923nnn4jZ9+vS8/PLLSZJvfOMbueKKK3LYYYflX/7lX/LMM88Ur/VP//RPmTp1ag444IBcfPHFmTlz5sb/IAAAAAAou549exbDV0nzXtDGeP7551NdXZ1DDjmkONa1a9fstddezXpjbdu2bdbnes9fj9XW1iZJ8cG+vx7blJqS5Ktf/WomTZpUPPfuu+/e4KsH37uPww47rNnYYYcdlnnz5q13dfkNKRQKH9i3/N3vfpe33347Rx55ZLMe24033ljssW2q1q1bZ9KkSXn11VczZsyY1NfX58orryyuDvZ+rr/++txwww258847i6Gs2bNn56WXXkrHjh2L9XXp0iVvv/12yTVuDAEsAAAA2EJ69+6dqqqqTQ43tWr17j/f/3q58r9eBv09bdq0abZfVVW13rG1a9d+4GeWIyS2KZ/drl27D7zeAQcckA996EOZNGnSepduf7/PW7t2bVq3bp3Zs2dnzpw5xe3555/PD3/4wyTvNrd+//vf54wzzsizzz6bgw8+OOPHj0+SDBkyJAsWLMjQoUPz+uuvZ+DAgRkxYsQH/xAAAAAA2GidOnXK0qVL1xl/44031lkNqdS+13vW1196b/yve2Pt2rVbb6/srz//vePrG9uUmpLkzDPPzO9///v893//d26++eb07Nmz+Hq9jan3vbFN1adPnw/sW753L3fffXezHtvvfve7/Od//ucmf+Zf+7u/+7ucccYZ+fGPf1wMev3kJz/Z4PxHH300F154YW688cZmq5OtXbs2Bx10ULP65syZkxdffDGnn376ZtX4fgSwAAAAYAvp0qVLjjrqqPz4xz/Om2++uc7xN954Y73nvff01l8/8TVnzpyy1fXEE0+ss99Sr0nckP322y8PPfTQ+875+7//+zzyyCO58847c+GFF27S9Q888MCsWbMmixcvzkc+8pFm218vvd6jR498/etfz2233Zbhw4fnZz/7WfHYhz70oZx99tm5+eabM27cuFx//fWbdpMAAAAAvK+PfvSjefLJJ9cZnzVrVvbaa6+yftbee++dd955J//v//2/4tif//znvPjii/nYxz5W1s9an7Zt2653haquXbvmxBNPzKRJkzJp0qR8+ctfft/r7L333pkxY0azsZkzZ6ZPnz7NVov/IKeffnoefPDB/Pa3v13n2DvvvJM333wze++9d2pqavLKK6+s02Pr0aPHRn/WB9l1113TvXv39fZQk+Sll17KKaeckssuuywnn3xys2Mf//jHM2/evHTr1m2dGjt37ly2Gv+WABYAAABsQdddd13WrFmTT37yk/nlL3+ZefPm5fnnn8+PfvSjdV4F+J73GhgNDQ158cUXc/fdd+eaa64pW02PP/54xowZkxdffDE//vGP8x//8R+56KKLynb9jTFy5MjMmjUr5513Xp555pn8z//8TyZOnJg//elPzeb16dMnjzzySPF1hBurT58++eIXv5gzzzwzt912W+bPn59Zs2blqquuyj333JMkGTp0aO6///7Mnz8/Tz31VB5++OFis+273/1u7rzzzrz00kuZO3dufvWrX22RRhwAAADAjuS8887Lyy+/nPPPPz9PP/10sV91ww035Fvf+lZZP6t379757Gc/m6997WuZMWNGnn766XzpS1/K3/3d3+Wzn/1sWT9rfXr27Jn58+dnzpw5+dOf/pSmpqbisa9+9auZMmVKnn/++Zx11lnve53hw4fnoYceyve///28+OKLmTJlSiZMmLDJq7cPHTo0hx12WAYOHJgf//jHefrpp/P73/8+//7v/55DDjkk8+bNS8eOHTNixIh885vfzJQpU/Lyyy/nt7/9bX784x9nypQpJf0cfvrTn+af/umf8sADD+Tll1/O3Llzc8kll2Tu3Lk5/vjj15m/cuXKHH/88TnggAPyf/7P/0ljY2NxS5IvfvGL2W233fLZz342v/71rzN//vxMnz49F110UV599dWSatwYFQ1gPfbYYzn++ONTX1+fqqqq3HHHHRuce+6556aqqirjxo1rNt7U1JQLL7wwu+22Wzp06JATTjihRX9gAAAAsDl69eqVp556KgMGDMjw4cPTt2/fHHnkkXnooYcyceLE9Z7Tpk2b3Hrrrfmf//mf7L///rnqqqtyxRVXlK2m4cOHZ/bs2TnwwAPz/e9/P9dcc02OOuqosl1/Y/Tp0ycPPPBAnn766Xzyk5/MoYcemjvvvDPV1dXrzN1rr73y8MMP59Zbb83w4cM3+jMmTZqUM888M8OHD89ee+2VE044If/v//2/4tN5a9asyfnnn5+PfexjOfroo7PXXnvluuuuS/LuE4kjR47Mfvvtl09/+tNp3bp1pk6dWp6bBwAAACDJu6GkX//613n55ZczePDgfOITn8jkyZMzefLkfO5znyv7502aNCkHHXRQjjvuuBx66KEpFAq555571nm9YUs45ZRTcvTRR2fAgAH50Ic+lFtvvbV4bNCgQenevXuOOuqo1NfXv+91Pv7xj+ff//3fM3Xq1PTt2zff/e53873vfS9nn332JtVTU1OTadOm5eKLL85Pf/rTfOpTn8onPvGJ/OhHP8o3vvGN9O3bN0ny/e9/P9/97nczevTofOxjH8tRRx2V//qv/0qvXr02+WeQJJ/85CezYsWKfP3rX88+++yTI444Ik888UTuuOOOHHHEEevM/+Mf/5j/+Z//ycMPP5z6+vp07969uCVJ+/bt89hjj2WPPfbIySefnI997GP5yle+kpUrV67zGstyqiqU8uLHMrn33nvz+OOP5+Mf/3hOOeWU3H777TnxxBPXmXfHHXekoaEh//u//5tvfetbzZ5w/ad/+qf813/9VyZPnpyuXbtm+PDh+ctf/pLZs2dv9FJqy5YtS+fOnbN06dIW/WG3pJ6X3l3pEnZof/jBsZUuAQAAdihvv/125s+fn169emWnnXaqdDnbtJ49e2bo0KGbtJoU7+/9fj+3hx4EbA22i+9SQ8u99oCN0LC00hUAAECL0j/btr311lupr6/P//2//3edV+xRfuXo5637GOkWNGTIkAwZMuR957z22mu54IILcv/99+fYY5uHXJYuXZobbrghN910UwYNGpQkufnmm9OjR488+OCDW/xpXQAAAAAAAAAAKMXatWvT2NiYa665Jp07d84JJ5xQ6ZLYSBUNYH2QtWvX5owzzsi3vvWt7LPPPuscnz17dlavXp3BgwcXx+rr69O3b9/MnDlzgwGspqamZu/OXLZsWfmLBwAAAAAAAACAjfTKK6+kV69e2X333TN58uRUV2/VsR7+ylb9N3XVVVeluro63/jGN9Z7vLGxMW3bts2uu+7abLy2tjaNjY0bvO7o0aNz+eWXl7VWAAAA2Bb94Q9/qHQJAAAAAECSnj17plAoVLoMStCq0gVsyOzZs/PDH/4wkydPTlVV1SadWygU3veckSNHZunSpcVt4cKFm1suAAAAAAAAAACwA9pqA1i//vWvs3jx4uyxxx6prq5OdXV1FixYkOHDh6dnz55Jkrq6uqxatSpLlixpdu7ixYtTW1u7wWvX1NSkU6dOzTYAAAAAAAAAAIBNtdUGsM4444w888wzmTNnTnGrr6/Pt771rdx///1JkoMOOiht2rTJtGnTiuctWrQozz33XPr161ep0gEAANhC1q5dW+kSYB1+LwEAAICthT4FfLByfE+qy1BHyVasWJGXXnqpuD9//vzMmTMnXbp0yR577JGuXbs2m9+mTZvU1dVlr732SpJ07tw555xzToYPH56uXbumS5cuGTFiRPbdd98MGjRoi94LAAAAW07btm3TqlWrvP766/nQhz6Utm3bbvLr66HcCoVCVq1alf/93/9Nq1at0rZt20qXBAAAAOyg9M/gg5Wzn1fRANaTTz6ZAQMGFPeHDRuWJDnrrLMyefLkjbrGtddem+rq6px66qlZuXJlBg4cmMmTJ6d169YtUTIAAABbgVatWqVXr15ZtGhRXn/99UqXA820b98+e+yxR1q12moXHgcAAAC2c/pnsPHK0c+raACrf//+KRQKGz3/D3/4wzpjO+20U8aPH5/x48eXsTIAAAC2dm3bts0ee+yRd955J2vWrKl0OZAkad26daqrqz1RCgAAAFSc/hl8sHL18yoawAIAAIDNUVVVlTZt2qRNmzaVLgUAAAAAtjr6Z7BlWAsfAAAAAAAAAACgRAJYAAAAAAAAAAAAJRLAAgAAAAAAAAAAKJEAFgAAAAAAAAAAQIkEsAAAAAAAAAAAAEokgAUAAAAAbJSJEydmv/32S6dOndKpU6cceuihuffee4vHzz777FRVVTXbPvWpTzW7RlNTUy688MLstttu6dChQ0444YS8+uqrW/pWAAAAAMpGAAsAAAAA2Ci77757fvCDH+TJJ5/Mk08+mc985jP57Gc/m7lz5xbnHH300Vm0aFFxu+eee5pdY+jQobn99tszderUzJgxIytWrMhxxx2XNWvWbOnbAQAAACiL6koXAAAAAABsG44//vhm+1deeWUmTpyYJ554Ivvss0+SpKamJnV1des9f+nSpbnhhhty0003ZdCgQUmSm2++OT169MiDDz6Yo446qmVvAAAAAKAFWAELAAAAANhka9asydSpU/Pmm2/m0EMPLY4/+uij6datW/r06ZOvfe1rWbx4cfHY7Nmzs3r16gwePLg4Vl9fn759+2bmzJkb/KympqYsW7as2QYAAACwtRDAAgAAAAA22rPPPpudd945NTU1+frXv57bb789e++9d5JkyJAh+fnPf56HH34411xzTWbNmpXPfOYzaWpqSpI0Njambdu22XXXXZtds7a2No2NjRv8zNGjR6dz587FrUePHi13gwAAAACbyCsIAQAAAICNttdee2XOnDl544038stf/jJnnXVWpk+fnr333junnXZacV7fvn1z8MEHZ88998zdd9+dk08+eYPXLBQKqaqq2uDxkSNHZtiwYcX9ZcuWCWEBAAAAWw0BLAAAAABgo7Vt2zYf+chHkiQHH3xwZs2alR/+8If56U9/us7c7t27Z88998y8efOSJHV1dVm1alWWLFnSbBWsxYsXp1+/fhv8zJqamtTU1JT5TgAAAADKwysIAQAAAICSFQqF4isG/9af//znLFy4MN27d0+SHHTQQWnTpk2mTZtWnLNo0aI899xz7xvAAgAAANiaWQELAAAAANgol112WYYMGZIePXpk+fLlmTp1ah599NHcd999WbFiRRoaGnLKKaeke/fu+cMf/pDLLrssu+22W0466aQkSefOnXPOOedk+PDh6dq1a7p06ZIRI0Zk3333zaBBgyp8dwAAAAClEcACgG1Yz0vvrnQJO7Q//ODYSpcAAABb1B//+MecccYZWbRoUTp37pz99tsv9913X4488sisXLkyzz77bG688ca88cYb6d69ewYMGJBf/OIX6dixY/Ea1157baqrq3Pqqadm5cqVGThwYCZPnpzWrVtX8M4AAAAASieABQAAAABslBtuuGGDx9q1a5f777//A6+x0047Zfz48Rk/fnw5SwMAAAComFaVLgAAAAAAAAAAAGBbJYAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJKhrAeuyxx3L88cenvr4+VVVVueOOO4rHVq9enUsuuST77rtvOnTokPr6+px55pl5/fXXm12jqakpF154YXbbbbd06NAhJ5xwQl599dUtfCcAAAAAAAAAAMCOqKIBrDfffDP7779/JkyYsM6xt956K0899VS+853v5Kmnnsptt92WF198MSeccEKzeUOHDs3tt9+eqVOnZsaMGVmxYkWOO+64rFmzZkvdBgAAAAAAAAAAsIOqruSHDxkyJEOGDFnvsc6dO2fatGnNxsaPH59PfvKTeeWVV7LHHntk6dKlueGGG3LTTTdl0KBBSZKbb745PXr0yIMPPpijjjqqxe8BAAAAAAAAAADYcVV0BaxNtXTp0lRVVWWXXXZJksyePTurV6/O4MGDi3Pq6+vTt2/fzJw5c4PXaWpqyrJly5ptAAAAAAAAAAAAm2qbCWC9/fbbufTSS3P66aenU6dOSZLGxsa0bds2u+66a7O5tbW1aWxs3OC1Ro8enc6dOxe3Hj16tGjtAAAAAAAAAADA9mmbCGCtXr06n//857N27dpcd911Hzi/UCikqqpqg8dHjhyZpUuXFreFCxeWs1wAAAAAAAAAAGAHsdUHsFavXp1TTz018+fPz7Rp04qrXyVJXV1dVq1alSVLljQ7Z/Hixamtrd3gNWtqatKpU6dmGwAAAAAAAAAAwKbaqgNY74Wv5s2blwcffDBdu3Ztdvyggw5KmzZtMm3atOLYokWL8txzz6Vfv35bulwAAAAAAAAAAGAHU13JD1+xYkVeeuml4v78+fMzZ86cdOnSJfX19fnHf/zHPPXUU/nVr36VNWvWpLGxMUnSpUuXtG3bNp07d84555yT4cOHp2vXrunSpUtGjBiRfffdN4MGDarUbQEAAAAAAAAAADuIigawnnzyyQwYMKC4P2zYsCTJWWedlYaGhtx1111JkgMOOKDZeY888kj69++fJLn22mtTXV2dU089NStXrszAgQMzefLktG7deovcAwAAAAAAAAAAsOOqaACrf//+KRQKGzz+fsfes9NOO2X8+PEZP358OUsDAAAAAAAAAAD4QK0qXQAAAAAAAAAAAMC2SgALAAAAAAAAAACgRAJYAAAAAAAAAAAAJRLAAgAAAAA2ysSJE7PffvulU6dO6dSpUw499NDce++9xeOFQiENDQ2pr69Pu3bt0r9//8ydO7fZNZqamnLhhRdmt912S4cOHXLCCSfk1Vdf3dK3AgAAAFA2AlgAAAAAwEbZfffd84Mf/CBPPvlknnzyyXzmM5/JZz/72WLIasyYMRk7dmwmTJiQWbNmpa6uLkceeWSWL19evMbQoUNz++23Z+rUqZkxY0ZWrFiR4447LmvWrKnUbQEAAABsFgEsAAAAAGCjHH/88TnmmGPSp0+f9OnTJ1deeWV23nnnPPHEEykUChk3bly+/e1v5+STT07fvn0zZcqUvPXWW7nllluSJEuXLs0NN9yQa665JoMGDcqBBx6Ym2++Oc8++2wefPDBCt8dAAAAQGkEsAAAAACATbZmzZpMnTo1b775Zg499NDMnz8/jY2NGTx4cHFOTU1NjjjiiMycOTNJMnv27KxevbrZnPr6+vTt27c4Z32ampqybNmyZhsAAADA1kIACwAAAADYaM8++2x23nnn1NTU5Otf/3puv/327L333mlsbEyS1NbWNptfW1tbPNbY2Ji2bdtm11133eCc9Rk9enQ6d+5c3Hr06FHmuwIAAAAonQAWAAAAALDR9tprr8yZMydPPPFE/umf/ilnnXVWfve73xWPV1VVNZtfKBTWGftbHzRn5MiRWbp0aXFbuHDh5t0EAAAAQBkJYAEAAAAAG61t27b5yEc+koMPPjijR4/O/vvvnx/+8Iepq6tLknVWslq8eHFxVay6urqsWrUqS5Ys2eCc9ampqUmnTp2abQAAAABbCwEsAAAAAKBkhUIhTU1N6dWrV+rq6jJt2rTisVWrVmX69Onp169fkuSggw5KmzZtms1ZtGhRnnvuueIcAAAAgG1NdaULAAAAAAC2DZdddlmGDBmSHj16ZPny5Zk6dWoeffTR3HfffamqqsrQoUMzatSo9O7dO717986oUaPSvn37nH766UmSzp0755xzzsnw4cPTtWvXdOnSJSNGjMi+++6bQYMGVfjuAAAAAEojgAUAAAAAbJQ//vGPOeOMM7Jo0aJ07tw5++23X+67774ceeSRSZKLL744K1euzHnnnZclS5bkkEMOyQMPPJCOHTsWr3Httdemuro6p556alauXJmBAwdm8uTJad26daVuCwAAAGCzCGABAAAAABvlhhtueN/jVVVVaWhoSENDwwbn7LTTThk/fnzGjx9f5uoAAAAAKqNVpQsAAAAAAAAAAADYVglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAAChRdaULANhcPS+9u9Il7ND+8INjK10CAAAAAAAAAFSMFbAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlKiiAazHHnssxx9/fOrr61NVVZU77rij2fFCoZCGhobU19enXbt26d+/f+bOndtsTlNTUy688MLstttu6dChQ0444YS8+uqrW/AuAAAAAAAAAACAHVVFA1hvvvlm9t9//0yYMGG9x8eMGZOxY8dmwoQJmTVrVurq6nLkkUdm+fLlxTlDhw7N7bffnqlTp2bGjBlZsWJFjjvuuKxZs2ZL3QYAAAAAAAAAALCDqq7khw8ZMiRDhgxZ77FCoZBx48bl29/+dk4++eQkyZQpU1JbW5tbbrkl5557bpYuXZobbrghN910UwYNGpQkufnmm9OjR488+OCDOeqoo7bYvQAAAAAAAAAAADueiq6A9X7mz5+fxsbGDB48uDhWU1OTI444IjNnzkySzJ49O6tXr242p76+Pn379i3OWZ+mpqYsW7as2QYAAAAAAAAAALCpttoAVmNjY5Kktra22XhtbW3xWGNjY9q2bZtdd911g3PWZ/To0encuXNx69GjR5mrBwAAAAAAAAAAdgRbbQDrPVVVVc32C4XCOmN/64PmjBw5MkuXLi1uCxcuLEutAAAAAAAAAADAjmWrDWDV1dUlyTorWS1evLi4KlZdXV1WrVqVJUuWbHDO+tTU1KRTp07NNgAAAAAAAAAAgE211QawevXqlbq6ukybNq04tmrVqkyfPj39+vVLkhx00EFp06ZNszmLFi3Kc889V5wDAAAAAAAAAADQUqor+eErVqzISy+9VNyfP39+5syZky5dumSPPfbI0KFDM2rUqPTu3Tu9e/fOqFGj0r59+5x++ulJks6dO+ecc87J8OHD07Vr13Tp0iUjRozIvvvum0GDBlXqtgAAAAAAAAAAgB1ERQNYTz75ZAYMGFDcHzZsWJLkrLPOyuTJk3PxxRdn5cqVOe+887JkyZIccsgheeCBB9KxY8fiOddee22qq6tz6qmnZuXKlRk4cGAmT56c1q1bb/H7AQAAAAAAAAAAdiwVDWD1798/hUJhg8erqqrS0NCQhoaGDc7ZaaedMn78+IwfP74FKgQAAAAAAAAAANiwVpUuAAAAAAAAAAAAYFslgAUAAAAAbJTRo0fnE5/4RDp27Jhu3brlxBNPzAsvvNBsztlnn52qqqpm26c+9almc5qamnLhhRdmt912S4cOHXLCCSfk1Vdf3ZK3AgAAAFA2AlgAAAAAwEaZPn16zj///DzxxBOZNm1a3nnnnQwePDhvvvlms3lHH310Fi1aVNzuueeeZseHDh2a22+/PVOnTs2MGTOyYsWKHHfccVmzZs2WvB0AAACAsqiudAEAAAAAwLbhvvvua7Y/adKkdOvWLbNnz86nP/3p4nhNTU3q6urWe42lS5fmhhtuyE033ZRBgwYlSW6++eb06NEjDz74YI466qiWuwEAAACAFmAFLAAAAACgJEuXLk2SdOnSpdn4o48+mm7duqVPnz752te+lsWLFxePzZ49O6tXr87gwYOLY/X19enbt29mzpy53s9pamrKsmXLmm0AAAAAWwsBLAAAAABgkxUKhQwbNiyHH354+vbtWxwfMmRIfv7zn+fhhx/ONddck1mzZuUzn/lMmpqakiSNjY1p27Ztdt1112bXq62tTWNj43o/a/To0encuXNx69GjR8vdGAAAAMAm8gpCAAAAAGCTXXDBBXnmmWcyY8aMZuOnnXZa8c99+/bNwQcfnD333DN33313Tj755A1er1AopKqqar3HRo4cmWHDhhX3ly1bJoQFAAAAbDWsgAUAAAAAbJILL7wwd911Vx555JHsvvvu7zu3e/fu2XPPPTNv3rwkSV1dXVatWpUlS5Y0m7d48eLU1tau9xo1NTXp1KlTsw0AAABgayGABQAAAABslEKhkAsuuCC33XZbHn744fTq1esDz/nzn/+chQsXpnv37kmSgw46KG3atMm0adOKcxYtWpTnnnsu/fr1a7HaAQAAAFqKVxACAAAAABvl/PPPzy233JI777wzHTt2TGNjY5Kkc+fOadeuXVasWJGGhoaccsop6d69e/7whz/ksssuy2677ZaTTjqpOPecc87J8OHD07Vr13Tp0iUjRozIvvvum0GDBlXy9gAAAABKIoAFAAAAAGyUiRMnJkn69+/fbHzSpEk5++yz07p16zz77LO58cYb88Ybb6R79+4ZMGBAfvGLX6Rjx47F+ddee22qq6tz6qmnZuXKlRk4cGAmT56c1q1bb8nbAQAAACgLASwAAAAAYKMUCoX3Pd6uXbvcf//9H3idnXbaKePHj8/48ePLVRoAAABAxbSqdAEAAAAAAAAAAADbqpICWPPnzy93HQAAAABAC9LTAwAAAGgZJQWwPvKRj2TAgAG5+eab8/bbb5e7JgAAAACgzPT0AAAAAFpGSQGsp59+OgceeGCGDx+eurq6nHvuufnNb35T7toAAAAAgDLR0wMAAABoGSUFsPr27ZuxY8fmtddey6RJk9LY2JjDDz88++yzT8aOHZv//d//LXedAAAAAMBm0NMDAAAAaBklBbDeU11dnZNOOin//u//nquuuiovv/xyRowYkd133z1nnnlmFi1aVK46AQAAAIAy0NMDAAAAKK/NCmA9+eSTOe+889K9e/eMHTs2I0aMyMsvv5yHH344r732Wj772c+Wq04AAAAAoAz09AAAAADKq7qUk8aOHZtJkyblhRdeyDHHHJMbb7wxxxxzTFq1ejfP1atXr/z0pz/NRz/60bIWCwAAAACURk8PAAAAoGWUFMCaOHFivvKVr+TLX/5y6urq1jtnjz32yA033LBZxQEAAAAA5aGnBwAAANAySgpgzZs37wPntG3bNmeddVYplwcAAAAAykxPDwAAAKBltCrlpEmTJuU//uM/1hn/j//4j0yZMmWziwIAAAAAyktPDwAAAKBllBTA+sEPfpDddtttnfFu3bpl1KhRm10UAAAAAFBeenoAAAAALaOkANaCBQvSq1evdcb33HPPvPLKK5tdFAAAAABQXnp6AAAAAC2jpABWt27d8swzz6wz/vTTT6dr166bXRQAAAAAUF56egAAAAAto6QA1uc///l84xvfyCOPPJI1a9ZkzZo1efjhh3PRRRfl85//fLlrBAAAAAA2k54eAAAAQMuoLuWkK664IgsWLMjAgQNTXf3uJdauXZszzzwzo0aNKmuBAAAAAMDm09MDAAAAaBklBbDatm2bX/ziF/n+97+fp59+Ou3atcu+++6bPffcs9z1AQAAAABloKcHAAAA0DJKCmC9p0+fPunTp0+5agEAAAAAWpieHgAAAEB5lRTAWrNmTSZPnpyHHnooixcvztq1a5sdf/jhh8tSHAAAAABQHnp6AAAAAC2jpADWRRddlMmTJ+fYY49N3759U1VVVe66AAAAAIAy0tMDAAAAaBklBbCmTp2af//3f88xxxxT7noAAAAAgBagpwcAAADQMlqVclLbtm3zkY98pNy1rOOdd97JP//zP6dXr15p165dPvzhD+d73/tes+XRC4VCGhoaUl9fn3bt2qV///6ZO3dui9cGAAAAANuSLdXTAwAAANjRlBTAGj58eH74wx+mUCiUu55mrrrqqvzkJz/JhAkT8vzzz2fMmDG5+uqrM378+OKcMWPGZOzYsZkwYUJmzZqVurq6HHnkkVm+fHmL1gYAAAAA25It1dMDAAAA2NGU9ArCGTNm5JFHHsm9996bffbZJ23atGl2/LbbbitLcf/93/+dz372szn22GOTJD179sytt96aJ598Msm7q1+NGzcu3/72t3PyyScnSaZMmZLa2trccsstOffcc8tSBwAAAABs67ZUTw8AAABgR1NSAGuXXXbJSSedVO5a1nH44YfnJz/5SV588cX06dMnTz/9dGbMmJFx48YlSebPn5/GxsYMHjy4eE5NTU2OOOKIzJw5c4MBrKampjQ1NRX3ly1b1qL3AQAAAACVtqV6egAAAAA7mpICWJMmTSp3Het1ySWXZOnSpfnoRz+a1q1bZ82aNbnyyivzhS98IUnS2NiYJKmtrW12Xm1tbRYsWLDB644ePTqXX355yxUOsAPpeendlS4BAACAjbClenoAAAAAO5pWpZ74zjvv5MEHH8xPf/rTLF++PEny+uuvZ8WKFWUr7he/+EVuvvnm3HLLLXnqqacyZcqU/Ou//mumTJnSbF5VVVWz/UKhsM7YXxs5cmSWLl1a3BYuXFi2mgEAAABga7UlenoAAAAAO5qSVsBasGBBjj766LzyyitpamrKkUcemY4dO2bMmDF5++2385Of/KQsxX3rW9/KpZdems9//vNJkn333TcLFizI6NGjc9ZZZ6Wuri7Juythde/evXje4sWL11kV66/V1NSkpqamLDUCAAAAwLZgS/X0AAAAAHY0Ja2AddFFF+Xggw/OkiVL0q5du+L4SSedlIceeqhsxb311ltp1ap5ia1bt87atWuTJL169UpdXV2mTZtWPL5q1apMnz49/fr1K1sdAAAAALCt21I9PQAAAIAdTUkrYM2YMSOPP/542rZt22x8zz33zGuvvVaWwpLk+OOPz5VXXpk99tgj++yzT377299m7Nix+cpXvpLk3VcPDh06NKNGjUrv3r3Tu3fvjBo1Ku3bt8/pp59etjoAAAAAYFu3pXp6AAAAADuakgJYa9euzZo1a9YZf/XVV9OxY8fNLuo948ePz3e+852cd955Wbx4cerr63Puuefmu9/9bnHOxRdfnJUrV+a8887LkiVLcsghh+SBBx4oax0AAAAAsK3bUj09AAAAgB1NSa8gPPLIIzNu3LjiflVVVVasWJF/+Zd/yTHHHFOu2tKxY8eMGzcuCxYsyMqVK/Pyyy/niiuuaPaUXlVVVRoaGrJo0aK8/fbbmT59evr27Vu2GgAAAABge1COnt7o0aPziU98Ih07dky3bt1y4okn5oUXXmg2p1AopKGhIfX19WnXrl369++fuXPnNpvT1NSUCy+8MLvttls6dOiQE044Ia+++upm3yMAAABAJZQUwLr22mszffr07L333nn77bdz+umnp2fPnnnttddy1VVXlbtGAAAAAGAzlaOnN3369Jx//vl54oknMm3atLzzzjsZPHhw3nzzzeKcMWPGZOzYsZkwYUJmzZqVurq6HHnkkVm+fHlxztChQ3P77bdn6tSpmTFjRlasWJHjjjtuvSt0AQAAAGztSnoFYX19febMmZNbb701Tz31VNauXZtzzjknX/ziF9OuXbty1wgAAAAAbKZy9PTuu+++ZvuTJk1Kt27dMnv27Hz6059OoVDIuHHj8u1vfzsnn3xykmTKlCmpra3NLbfcknPPPTdLly7NDTfckJtuuimDBg1Kktx8883p0aNHHnzwwRx11FHlvXEAAACAFlZSACtJ2rVrl6985Sv5yle+Us56AAAAAIAWUu6e3tKlS5MkXbp0SZLMnz8/jY2NGTx4cHFOTU1NjjjiiMycOTPnnntuZs+endWrVzebU19fn759+2bmzJnrDWA1NTWlqampuL9s2bKy1A8AAABQDiUFsG688cb3PX7mmWeWVAwAAAAA0DLK3dMrFAoZNmxYDj/88PTt2zdJ0tjYmCSpra1tNre2tjYLFiwozmnbtm123XXXdea8d/7fGj16dC6//PJNqg8AAABgSykpgHXRRRc121+9enXeeuuttG3bNu3btxfAAgAAAICtTLl7ehdccEGeeeaZzJgxY51jVVVVzfYLhcI6Y3/r/eaMHDkyw4YNK+4vW7YsPXr02KR6AQAAAFpKq1JOWrJkSbNtxYoVeeGFF3L44Yfn1ltvLXeNAAAAAMBmKmdP78ILL8xdd92VRx55JLvvvntxvK6uLknWWclq8eLFxVWx6urqsmrVqixZsmSDc/5WTU1NOnXq1GwDAAAA2FqUFMBan969e+cHP/jBOk/SAQAAAABbp03t6RUKhVxwwQW57bbb8vDDD6dXr17Njvfq1St1dXWZNm1acWzVqlWZPn16+vXrlyQ56KCD0qZNm2ZzFi1alOeee644BwAAAGBbUtIrCDekdevWef3118t5SQAAAACgBW1KT+/888/PLbfckjvvvDMdO3YsrnTVuXPntGvXLlVVVRk6dGhGjRqV3r17p3fv3hk1alTat2+f008/vTj3nHPOyfDhw9O1a9d06dIlI0aMyL777ptBgwa12H0CAAAAtJSSAlh33XVXs/1CoZBFixZlwoQJOeyww8pSGAAAAABQPuXo6U2cODFJ0r9//2bjkyZNytlnn50kufjii7Ny5cqcd955WbJkSQ455JA88MAD6dixY3H+tddem+rq6px66qlZuXJlBg4cmMmTJ6d169al3yAAAABAhZQUwDrxxBOb7VdVVeVDH/pQPvOZz+Saa64pR10AAAAAQBmVo6dXKBQ+cE5VVVUaGhrS0NCwwTk77bRTxo8fn/Hjx2/U5wIAAABszUoKYK1du7bcdQAAAAAALUhPDwAAAKBltKp0AQAAAAAAAAAAANuqklbAGjZs2EbPHTt2bCkfAQAAAACUkZ4eAAAAQMsoKYD129/+Nk899VTeeeed7LXXXkmSF198Ma1bt87HP/7x4ryqqqryVAkAAAAAbBY9PQAAAICWUVIA6/jjj0/Hjh0zZcqU7LrrrkmSJUuW5Mtf/nL+4R/+IcOHDy9rkQAAAADA5tHTAwAAAGgZrUo56Zprrsno0aOLjZok2XXXXXPFFVfkmmuuKVtxAAAAAEB56OkBAAAAtIySAljLli3LH//4x3XGFy9enOXLl292UQAAAABAeenpAQAAALSMkl5BeNJJJ+XLX/5yrrnmmnzqU59KkjzxxBP51re+lZNPPrmsBQIAAMDWpueld1e6hB3aH35wbKVLgG2Snh4AAABAyygpgPWTn/wkI0aMyJe+9KWsXr363QtVV+ecc87J1VdfXdYCAQAAAIDNp6cHAAAA0DJKCmC1b98+1113Xa6++uq8/PLLKRQK+chHPpIOHTqUuz4AAAAAoAz09AAAAABaRqvNOXnRokVZtGhR+vTpkw4dOqRQKJSrLgAAAACgBejpAQAAAJRXSQGsP//5zxk4cGD69OmTY445JosWLUqSfPWrX83w4cPLWiAAAAAAsPn09AAAAABaRkkBrG9+85tp06ZNXnnllbRv3744ftppp+W+++4rW3EAAAAAQHno6QEAAAC0jOpSTnrggQdy//33Z/fdd2823rt37yxYsKAshQEAAAAA5aOnBwAAANAySloB680332z2lNx7/vSnP6WmpmaziwIAAAAAyktPDwAAAKBllBTA+vSnP50bb7yxuF9VVZW1a9fm6quvzoABA8pWHAAAAABQHnp6AAAAAC2jpFcQXn311enfv3+efPLJrFq1KhdffHHmzp2bv/zlL3n88cfLXSMAAAAAsJn09AAAAABaRkkrYO2999555pln8slPfjJHHnlk3nzzzZx88sn57W9/m7//+78vd40AAAAAwGbS0wMAAABoGZu8Atbq1aszePDg/PSnP83ll1/eEjUBAAAAAGWkpwcAAADQcjZ5Baw2bdrkueeeS1VVVUvUAwAAAACUmZ4eAAAAQMsp6RWEZ555Zm644YZy1wIAAAAAtBA9PQAAAICWscmvIEySVatW5d/+7d8ybdq0HHzwwenQoUOz42PHji1LcQAAAABAeejpAQAAALSMTQpg/f73v0/Pnj3z3HPP5eMf/3iS5MUXX2w2p9zLmL/22mu55JJLcu+992blypXp06dPbrjhhhx00EFJkkKhkMsvvzzXX399lixZkkMOOSQ//vGPs88++5S1DgAAAADYFlWipwcAAACwI9mkAFbv3r2zaNGiPPLII0mS0047LT/60Y9SW1vbIsUtWbIkhx12WAYMGJB777033bp1y8svv5xddtmlOGfMmDEZO3ZsJk+enD59+uSKK67IkUcemRdeeCEdO3ZskboAAAAAYFuxpXt6AAAAADuaTQpgFQqFZvv33ntv3nzzzbIW9Neuuuqq9OjRI5MmTSqO9ezZs1k948aNy7e//e2cfPLJSZIpU6aktrY2t9xyS84999wWqw0AAAAAtgVbuqcHAAAAsKNptTkn/23zptzuuuuuHHzwwfnc5z6Xbt265cADD8zPfvaz4vH58+ensbExgwcPLo7V1NTkiCOOyMyZMzd43aampixbtqzZBgAAAAA7gpbu6QEAAADsaDYpgFVVVZWqqqp1xlrK73//+0ycODG9e/fO/fffn69//ev5xje+kRtvvDFJ0tjYmCTrLJdeW1tbPLY+o0ePTufOnYtbjx49WuweAAAAAKCStnRPDwAAAGBHs8mvIDz77LNTU1OTJHn77bfz9a9/PR06dGg277bbbitLcWvXrs3BBx+cUaNGJUkOPPDAzJ07NxMnTsyZZ55ZnPe3DaNCofC+TaSRI0dm2LBhxf1ly5YJYQEAAACwXdrSPT0AAACAHc0mBbDOOuusZvtf+tKXylrM3+revXv23nvvZmMf+9jH8stf/jJJUldXl+TdlbC6d+9enLN48eJ1VsX6azU1NcWGEwAAAABsz7Z0Tw8AAABgR7NJAaxJkya1VB3rddhhh+WFF15oNvbiiy9mzz33TJL06tUrdXV1mTZtWg488MAkyapVqzJ9+vRcddVVW7RWAAAAANgabemeHgAAAMCOplWlC3g/3/zmN/PEE09k1KhReemll3LLLbfk+uuvz/nnn5/k3VcPDh06NKNGjcrtt9+e5557LmeffXbat2+f008/vcLVAwAAAMD25bHHHsvxxx+f+vr6VFVV5Y477mh2/Oyzz05VVVWz7VOf+lSzOU1NTbnwwguz2267pUOHDjnhhBPy6quvbsG7AAAAACivrTqA9YlPfCK33357br311vTt2zff//73M27cuHzxi18szrn44oszdOjQnHfeeTn44IPz2muv5YEHHkjHjh0rWDkAAAAAbH/efPPN7L///pkwYcIG5xx99NFZtGhRcbvnnnuaHR86dGhuv/32TJ06NTNmzMiKFSty3HHHZc2aNS1dPgAAAECL2KRXEFbCcccdl+OOO26Dx6uqqtLQ0JCGhoYtVxQAAAAA7ICGDBmSIUOGvO+cmpqa1NXVrffY0qVLc8MNN+Smm27KoEGDkiQ333xzevTokQcffDBHHXVU2WsGAAAAaGlb9QpYAAAAAMC25dFHH023bt3Sp0+ffO1rX8vixYuLx2bPnp3Vq1dn8ODBxbH6+vr07ds3M2fO3OA1m5qasmzZsmYbAAAAwNZCAAsAAAAAKIshQ4bk5z//eR5++OFcc801mTVrVj7zmc+kqakpSdLY2Ji2bdtm1113bXZebW1tGhsbN3jd0aNHp3PnzsWtR48eLXofAAAAAJtiq38FIQAAAACwbTjttNOKf+7bt28OPvjg7Lnnnrn77rtz8sknb/C8QqGQqqqqDR4fOXJkhg0bVtxftmyZEBYAAACw1bACFgAAAADQIrp3754999wz8+bNS5LU1dVl1apVWbJkSbN5ixcvTm1t7QavU1NTk06dOjXbAAAAALYWAlgAAAAAQIv485//nIULF6Z79+5JkoMOOiht2rTJtGnTinMWLVqU5557Lv369atUmQAAAACbxSsIAQAAAICNsmLFirz00kvF/fnz52fOnDnp0qVLunTpkoaGhpxyyinp3r17/vCHP+Syyy7LbrvtlpNOOilJ0rlz55xzzjkZPnx4unbtmi5dumTEiBHZd999M2jQoErd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qa+vT69evfL5z38+v//975NY0R4AAADYPm0TAawLL7wwd911Vx555JHsvvvuxfH3nqJ7byWs9yxevHidVbH+Wk1NTTp16tRsAwAAAAA23yGHHJIbb7wx999/f372s5+lsbEx/fr1y5///Of3XdH+b3t8f2306NHp3LlzcevRo0eL3gMAAADAptiqA1iFQiEXXHBBbrvttjz88MPp1atXs+O9evVKXV1dpk2bVhxbtWpVpk+fnn79+m3pcgEAAABghzdkyJCccsop2XfffTNo0KDcfffdSd591eB7rGgPAAAAbE+qK13A+zn//PNzyy235M4770zHjh2LT8F17tw57dq1S1VVVYYOHZpRo0ald+/e6d27d0aNGpX27dvn9NNPr3D1AAAAAECHDh2y7777Zt68eTnxxBOTvLuifffu3YtzNmZF+5qampYuFQAAAKAkW/UKWBMnTszSpUvTv3//dO/evbj94he/KM65+OKLM3To0Jx33nk5+OCD89prr+WBBx5Ix44dK1g5AAAAAJAkTU1Nef7559O9e3cr2gMAAADbpa16BaxCofCBc6qqqtLQ0JCGhoaWLwgAAAAAeF8jRozI8ccfnz322COLFy/OFVdckWXLluWss86yoj1bj4bOla4AAACA7chWHcACAAAAALYtr776ar7whS/kT3/6Uz70oQ/lU5/6VJ544onsueeeSd5d0X7lypU577zzsmTJkhxyyCFWtAcAAAC2aQJYAAAAAEDZTJ069X2PW9EeAAAA2N60qnQBAAAAAAAAAAAA2yorYAEAAAAAAOwoGjpXuoIdW8PSSlcAAEALsAIWAAAAAAAAAABAiQSwAAAAAAAAAAAASuQVhAAAANugnpfeXekSAAAAAACAWAELAAAAAAAAAACgZAJYAAAAAAAAAAAAJRLAAgAAAAAAAAAAKJEAFgAAAAAAAAAAQImqK10AAAAAAAAA7BAaOle6gh1bw9JKVwAAbKcEsAAAAAAAAABgeycEWllCoLBd8wpCAAAAAAAAAACAElkBCwAAAAAAANj+Wf2n8qwABMB2ygpYAAAAAAAAAAAAJRLAAgAAAAAAAAAAKJEAFgAAAAAAAAAAQIkEsAAAAAAAAAAAAEokgAUAAAAAAAAAAFAiASwAAAAAAAAAAIASVVe6AAAAAAAAAAB2AA2dK10BALQIK2ABAAAAAAAAAACUSAALAAAAAAAAAACgRAJYAAAAAAAAAAAAJRLAAgAAAAAAAAAAKJEAFgAAAAAAAAAAQIkEsAAAAAAAAAAAAEokgAUAAAAAAAAAAFAiASwAAAAAAAAAAIASCWABAAAAAAAAAACUqLrSBQAAAAAAAAAAbNcaOle6AqAFbTcrYF133XXp1atXdtpppxx00EH59a9/XemSAAAAAIAN0M8DAAAAthfbRQDrF7/4RYYOHZpvf/vb+e1vf5t/+Id/yJAhQ/LKK69UujQAAAAA4G/o5wEAAADbk+0igDV27Nicc845+epXv5qPfexjGTduXHr06JGJEydWujQAAAAA4G/o5wEAAADbk+pKF7C5Vq1aldmzZ+fSSy9tNj548ODMnDlzvec0NTWlqampuL906dIkybJly1qu0Ba2tumtSpcAAAAAW8S2/O/392ovFAoVrgQqRz/v/6/J/w4AAACwg9iG//2+sf28bT6A9ac//Slr1qxJbW1ts/Ha2to0Njau95zRo0fn8ssvX2e8R48eLVIjAAAAUD6dx1W6gs23fPnydO7cudJlQEXo5wEAAMAO5gfbfh/sg/p523wA6z1VVVXN9guFwjpj7xk5cmSGDRtW3F+7dm3+8pe/pGvXrhs8B8ph2bJl6dGjRxYuXJhOnTpVuhzY5vlOQfn5XkF5+U5B+W3r36tCoZDly5envr6+0qVAxennsS3Y1v+7A1sb3ykoP98rKC/fKSi/bf17tbH9vG0+gLXbbruldevW6zwdt3jx4nWeontPTU1Nampqmo3tsssuLVUirKNTp07b5P+wwNbKdwrKz/cKyst3CspvW/5eWfmKHZ1+Htuibfm/O7A18p2C8vO9gvLynYLy25a/VxvTz2u1BepoUW3bts1BBx2UadOmNRufNm1a+vXrV6GqAAAAAID10c8DAAAAtjfb/ApYSTJs2LCcccYZOfjgg3PooYfm+uuvzyuvvJKvf/3rlS4NAAAAAPgb+nkAAADA9mS7CGCddtpp+fOf/5zvfe97WbRoUfr27Zt77rkne+65Z6VLg2ZqamryL//yL+ssmQ+UxncKys/3CsrLdwrKz/cKtg/6eWwr/HcHyst3CsrP9wrKy3cKym9H+V5VFQqFQqWLAAAAAAAAAAAA2Ba1qnQBAAAAAAAAAAAA2yoBLAAAAAAAAAAAgBIJYAEAAAAAAAAAAJRIAAsAAAAAAAAAAKBEAljQwkaPHp1PfOIT6dixY7p165YTTzwxL7zwQqXLgu3G6NGjU1VVlaFDh1a6FNimvfbaa/nSl76Url27pn379jnggAMye/bsSpcF26x33nkn//zP/5xevXqlXbt2+fCHP5zvfe97Wbt2baVLg23CY489luOPPz719fWpqqrKHXfc0ex4oVBIQ0ND6uvr065du/Tv3z9z586tTLEAbHf086Dl6enB5tPPg/LSz4PNt6P39ASwoIVNnz49559/fp544olMmzYt77zzTgYPHpw333yz0qXBNm/WrFm5/vrrs99++1W6FNimLVmyJIcddljatGmTe++9N7/73e9yzTXXZJdddql0abDNuuqqq/KTn/wkEyZMyPPPP58xY8bk6quvzvjx4ytdGmwT3nzzzey///6ZMGHCeo+PGTMmY8eOzYQJEzJr1qzU1dXlyCOPzPLly7dwpQBsj/TzoGXp6cHm08+D8tPPg823o/f0qgqFQqHSRcCO5H//93/TrVu3TJ8+PZ/+9KcrXQ5ss1asWJGPf/zjue6663LFFVfkgAMOyLhx4ypdFmyTLr300jz++OP59a9/XelSYLtx3HHHpba2NjfccENx7JRTTkn79u1z0003VbAy2PZUVVXl9ttvz4knnpjk3Sfl6uvrM3To0FxyySVJkqamptTW1uaqq67KueeeW8FqAdge6edB+ejpQXno50H56edBee2IPT0rYMEWtnTp0iRJly5dKlwJbNvOP//8HHvssRk0aFClS4Ft3l133ZWDDz44n/vc59KtW7cceOCB+dnPflbpsmCbdvjhh+ehhx7Kiy++mCR5+umnM2PGjBxzzDEVrgy2ffPnz09jY2MGDx5cHKupqckRRxyRmTNnVrAyALZX+nlQPnp6UB76eVB++nnQsnaEnl51pQuAHUmhUMiwYcNy+OGHp2/fvpUuB7ZZU6dOzVNPPZVZs2ZVuhTYLvz+97/PxIkTM2zYsFx22WX5zW9+k2984xupqanJmWeeWenyYJt0ySWXZOnSpfnoRz+a1q1bZ82aNbnyyivzhS98odKlwTavsbExSVJbW9tsvLa2NgsWLKhESQBsx/TzoHz09KB89POg/PTzoGXtCD09ASzYgi644II888wzmTFjRqVLgW3WwoULc9FFF+WBBx7ITjvtVOlyYLuwdu3aHHzwwRk1alSS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1fdfv359WrVq9ZbjXt+T3J6ef/75/OIXv8gll1ySadOmbbLo0c6itra23CW8pbKugHXqqafm5JNPTs+ePdOzZ8987nOfy1577dUkdVdZWZna2tri9s//YjY0NGTatGmZPHlyBg0alMMOOyy33nprFi5cmAceeKAcUwIAAKBMXls96pZbbsn++++f6urqfOADH8iqVauKY/55Na0BAwZk8uTJeeSRR1JRUZEBAwYkSVasWJHzzz8/++yzT9q2bZuhQ4dm8eLFxXvMmDEje++9d37wgx/k4IMPTmVlZZ577rnsv//+ueaaa3L++ednr732Srdu3fK9730vf/3rX3P66adnr732St++ffPYY49tcq/X/vrqq6/Ob37zm+Iq0DNmzMhHPvKRDBs2rMlcX3311dTW1uab3/zmG/79+O53v5vevXunsrIy+++/fyZPntzk78Pm5v56119/fR555JH85Cc/ycUXX5xDDz0073znO3POOefkl7/8ZXr06JEkKRQKmTRpUt75znemTZs2OeSQQ/J///d/b/nP7I3Mnj07dXV1mTRpUvr06ZN3vetdOemkk/KNb3xjkxezfvzjH6dXr17Za6+9ctJJJ2XZsmXFc/PmzcvgwYPTqVOnVFdX57jjjsuvf/3rJtdXVFTkxhtvzNChQ9OmTZt079493/nOd5qM+dOf/pSzzz47++yzTzp27JjTTz89zz77bMnzAwAAAGDbO/zww7PffvvlzjvvLB67884707Vr1xx22GFNxt5333055phjsvfee6djx44ZNmxYnnnmmeL5Z599NhUVFfn2t7+dAQMGZM8998ytt96aV199NZ/4xCeK111++eUZOXJkk0/lvX6F//333z8TJkzIRz7ykVRVVWW//fbLTTfd1KSeyy+/PD179kzbtm3zzne+M5/5zGdKWo1++vTpGTZsWD7+8Y/njjvuyJo1a5qcX7x4cY499tjsueeeOfjggzN79uzN3uePf/xjjj/++LRt2zaHHHJIfvGLXzQ5P3fu3Bx77LFp06ZNunbtmk984hNNnvVWc163bl0uueSS1NXVZc8998z++++fiRMnFs9XVFTk7rvvLu4vXLgwJ5xwQtq0aZOOHTvm3//937N69eri+VGjRuWMM87I//7v/6auri4dO3bMxRdfvE1X9C9rAOufbdiwIbNmzcqaNWvynve8p3j8oYceSufOndOzZ8989KMfzfLly4vn5s+fn/Xr12fIkCHFY126dEmfPn0yd+7cN3xWY2NjVq5c2WQDAABg5/fMM8/k7rvvzg9+8IP84Ac/yMMPP5zPf/7zmx1755135qMf/Wje8573ZNmyZcVGzKhRo/LYY4/lnnvuyS9+8YsUCoWcfPLJTf5w/vLLL2fixIn5xje+kSeffDKdO3dOklx33XV573vfm8cffzynnHJKzjvvvJx//vk599xz8+tf/zoHHHBAzj///BQKhU3qOfvsszN27Nj07t07y5Yty7Jly3L22Wfn3/7t33Lfffc1CRb98Ic/zOrVqzNixIjNzm3+/PkZMWJEPvCBD2ThwoUZP358PvOZz2TGjBlvOvfXu+2224ovPL1eq1at0q5duyTJf/7nf2b69Om58cYb8+STT+ZTn/pUzj333Dz88MObve9bqa2tzbJly/LII4+86biXX345//u//5tbbrkljzzySJ5//vmMGzeueH7VqlUZOXJkfvazn+XRRx9Njx49cvLJJzcJ5SXJZz7zmZx11ln5zW9+k3PPPTcf/OAHs2jRouIzjj/++Oy111555JFHMmfOnGLYa926dSXND3YVEydOTEVFRZMmcqFQyPjx49OlS5e0adMmAwYMyJNPPtnkusbGxlx66aXp1KlT2rVrl9NOOy0vvPDCdq4eAACA3cGHP/zhJl9d++Y3v5mPfOQjm4xbs2ZNxowZk3nz5uUnP/lJ9thjj5x55pmbrMZ++eWX5xOf+EQWLVqUE088Mddee21uu+22TJ8+PT//+c+zcuXKJkGhNzJ58uT069cvjz/+eC666KJ8/OMfz+9///vi+aqqqsyYMSO/+93v8sUvfjFf//rXc911172tuRcKhUyfPj3nnntuDjrooPTs2TPf/va3i+c3btyY4cOHFz9P+NWvfjWXX375Zu911VVXZdy4cVmwYEF69uyZD37wg3n11VeT/CMMdeKJJ2b48OH57W9/mzvuuCNz5szJJZdcssVz/tKXvpR77rkn3/72t/PUU0/l1ltvzf7777/ZWl5++eWcdNJJ2WeffTJv3rx85zvfyQMPPLDJ837605/mmWeeyU9/+tPMnDkzM2bMKPZHt4WyB7AWLlyYvfbaK5WVlfnYxz6Wu+66KwcffHCSZOjQobntttvy4IMPZvLkyZk3b15OOOGENDY2Jknq6+vTunXr7LPPPk3uWVNTk/r6+jd85sSJE1NdXV3cunbtuu0mCAAAwHazcePGzJgxI3369Mn73ve+nHfeefnJT36y2bEdOnRI27Zt07p16+KKy4sXL84999yTb3zjG3nf+96XQw45JLfddlv+9Kc/NWmcrF+/Pl/5ylfSv3//HHjggcUg0sknn5wLL7wwPXr0yH/9139l1apVOfLII/P+978/PXv2zOWXX55FixblL3/5yyb1tGnTJnvttVdatmxZXAW6TZs2xWfccsstxbHTp0/P+9///uy1116bnduUKVMycODAfOYzn0nPnj0zatSoXHLJJfnCF77whnPfnMWLF+eggw5607/na9asyZQpU/LNb34zJ554Yt75zndm1KhROffcc/O1r33tTa99I+9///vzwQ9+MMcdd1zq6upy5plnZurUqZu8QLV+/fp89atfTb9+/XL44YfnkksuafLP+4QTTsi5556bXr16pVevXvna176Wl19+eZNg2Pvf//7827/9W3r27JnPfvaz6devX2644YYkyaxZs7LHHnvkG9/4Rvr27ZtevXpl+vTpef755/PQQw+VND/YFcybNy833XRT3v3udzc5PmnSpEyZMiVTp07NvHnzUltbm8GDBzcJPo4ePTp33XVXZs2alTlz5mT16tUZNmxYNmzYsL2nAQAAwC7uvPPOy5w5c/Lss8/mueeey89//vOce+65m4w766yzMnz48PTo0SOHHnpopk2bloULF+Z3v/tdk3GjR4/O8OHD071793Tp0iU33HBDrrjiipx55pk56KCDMnXq1OKK92/m5JNPzkUXXZQDDjggl19+eTp16tSk1/Sf//mf6d+/f/bff/+ceuqpGTt2bJPw1JZ44IEH8vLLL+fEE09Mkpx77rmZNm1ak/OLFi3KLbfckkMPPTTHHntsJkyYsNl7jRs3Lqecckp69uyZq6++Os8991z+8Ic/JEm+8IUv5Jxzzsno0aPTo0eP9O/fP1/60pdy880355VXXtmiOT///PPp0aNHjjnmmHTr1i3HHHNMPvjBD262lttuuy1r167NzTffnD59+uSEE07I1KlTc8sttzTpu+6zzz6ZOnVqDjrooAwbNiynnHLKG/aKm0PZA1gHHnhgFixYkEcffTQf//jHM3LkyOK/wGeffXZOOeWU9OnTJ6eeemp+9KMf5emnn8699977pvcsFAqpqKh4w/NXXHFFGhoaitvSpUubdU4AAACUx/7775+qqqrifl1dXZOVlN/KokWL0rJlyxx11FHFYx07dsyBBx5YXBEpSVq3br1J6CBJk2M1NTVJkr59+25y7O3UlCT/9m//VnxTb/ny5bn33ns3+6beP8/jve99b5Nj733ve7N48eK3FXB4qz9fJ8nvfve7vPLKKxk8eHD22muv4nbzzTc3Wab97WjRokWmT5+eF154IZMmTUqXLl3yuc99rrg62Gvatm2bd73rXcX91//zXr58eT72sY+lZ8+exZewVq9eneeff77J8/55Je7X9l/75z1//vz84Q9/SFVVVXFuHTp0yCuvvFLy/GBnt3r16nzoQx/K17/+9SYvRhYKhVx//fW56qqrMnz48PTp0yczZ87Myy+/nNtvvz1J0tDQkGnTpmXy5MnFFfZuvfXWLFy4MA888EC5pgQAAMAuqlOnTjnllFMyc+bMTJ8+Paeccko6deq0ybhnnnkm55xzTt75znemffv26d69e5Js0kfq169f8a8bGhryl7/8Jf/6r/9aPNaiRYscccQRb1nXP/cRKyoqUltb26Sv9X//93855phjUltbm7322iuf+cxnNqnlrUybNi1nn312WrZsmST54Ac/mF/+8pd56qmnkvyjh7jffvtl3333LV7z+j7Z5uqtq6tL8v/1OOfPn58ZM2Y06Q2eeOKJ2bhxY5YsWbJFcx41alQWLFiQAw88MJ/4xCdy//33v+G8Fi1alEMOOaT4Umzyj97nxo0bi3NLkt69e6dFixZN6n67fdm3o+wBrNatW+eAAw5Iv379MnHixBxyyCH54he/uNmxdXV16datWxYvXpzkH58kWLduXVasWNFk3PLly4tN7c2prKxM+/btm2wAAADsmNq3b5+GhoZNjr/00kub/HmuVatWTfYrKio2WSb8zWzu04CvHf/nIFKbNm02G0z65+e/dn5zx95OTUly/vnn549//GN+8YtfFJffft/73vem83h9fW80tzfTs2fPJsGzzXltLvfee28WLFhQ3H73u9/l//7v/972M//Zv/zLv+S8887Ll7/85WLQ66tf/Wrx/Ob+ef/zPEeNGpX58+fn+uuvz9y5c7NgwYJ07Nhxiz4d+M//rI444ogmc1uwYEGefvrpnHPOOVs1P9hZXXzxxTnllFMyaNCgJseXLFmS+vr6DBkypHissrIyxx13XObOnZvkH03Z9evXNxnTpUuX9OnTpzhmcxobG7Ny5comGwAAAGyJj3zkI5kxY0Zmzpz5hi81nnrqqXnxxRfz9a9/Pb/85S/zy1/+Mkk26SP9c+jnNaX04d6sj/noo4/mAx/4QIYOHZof/OAHefzxx3PVVVdtUU/rNX//+99z99135ytf+UpatmyZli1b5l/+5V/y6quv5pvf/OYb1vlGL2O+WY9z48aNufDCC5v0zn7zm99k8eLFTV6efLM5H3744VmyZEk++9nPZu3atRkxYkT+3//7f5ut5c1eGv3n41vbK367yh7Aer1CoVD8xODrvfjii1m6dGkxTXfEEUekVatWmT17dnHMsmXL8sQTT6R///7bpV4AAAC2rYMOOiiPPfbYJsfnzZuXAw88sFmfdfDBB+fVV18tNliSf/xZ9Omnn06vXr2a9Vmb07p1682uUNWxY8ecccYZmT59eqZPn54Pf/jDb3qfgw8+OHPmzGlybO7cuenZs2eTt77eyjnnnJMHHnggjz/++CbnXn311axZsyYHH3xwKisr8/zzz+eAAw5osnXt2nWLn/VW9tlnn9TV1WXNmjVbfM3PfvazfOITn8jJJ5+c3r17p7KyMn/72982Gffoo49usv/apxcPP/zwLF68OJ07d95kftXV1Vs3KdgJzZo1K7/+9a8zceLETc7V19cnySYvRtbU1BTP1dfXp3Xr1k1Wznr9mM2ZOHFicSW76urqZv3fFwAAAHZtJ510UtatW5d169YVP8f3z1588cUsWrQo//mf/5mBAwemV69emywEtDnV1dWpqanJr371q+KxDRs2bLaX9nb8/Oc/T7du3XLVVVelX79+6dGjR5577rm3dY/bbrst++67b37zm980CUZdf/31mTlzZl599dUcfPDBef755/PnP/+5eN0vfvGLt13v4YcfnieffHKT3tkBBxyQ1q1bb/F92rdvn7PPPjtf//rXc8cdd+S73/1u/v73v28y7uCDD86CBQua9Al//vOfZ4899kjPnj3fdv3NpWXZnpzkyiuvzNChQ9O1a9esWrUqs2bNykMPPZT77rsvq1evzvjx43PWWWelrq4uzz77bK688sp06tQpZ555ZpJ//Mt8wQUXZOzYsenYsWM6dOiQcePGpW/fvpu8gQcAAMDO6aKLLsrUqVNz8cUX59///d/Tpk2bzJ49O9OmTcstt9zSrM/q0aNHTj/99Hz0ox/N1772tVRVVeXTn/50/uVf/iWnn356sz5rc/bff/8sWbIkCxYsyL777puqqqpUVlYm+cdnCIcNG5YNGzZk5MiRb3qfsWPH5sgjj8xnP/vZnH322fnFL36RqVOn5itf+crbqmf06NG59957M3DgwHz2s5/NMccck6qqqjz22GO59tprM23atBx66KEZN25cPvWpT2Xjxo055phjsnLlysydOzd77bXXW9a6OV/72teyYMGCnHnmmXnXu96VV155JTfffHOefPLJ3HDDDVt8nwMOOCC33HJL+vXrl5UrV+Y//uM/0qZNm03Gfec730m/fv1yzDHH5LbbbsuvfvWrTJs2LUnyoQ99KF/4whdy+umn53/+53+y77775vnnn8+dd96Z//iP/2iyRDvs6pYuXZpPfvKTuf/++7Pnnnu+4bjNvfn7Vp8zfasxV1xxRcaMGVPcX7lypRAWAAAAW6RFixbFVd4393LiPvvsk44dO+amm25KXV1dnn/++Xz605/eontfeumlmThxYg444IAcdNBBueGGG7JixYq3/HPwmznggAPy/PPPZ9asWTnyyCNz77335q677npb95g2bVr+3//7f+nTp0+T4926dcvll1+ee++9N6eeemoOPPDAnH/++Zk8eXJWrlyZq6666m3Xe/nll+foo4/OxRdfnI9+9KNp165dFi1alNmzZ29xL++6665LXV1dDj300Oyxxx75zne+k9ra2uy9996bjP3Qhz6U//7v/87IkSMzfvz4/PWvf82ll16a8847702/lretlTWA9Ze//CXnnXdeli1blurq6rz73e/Offfdl8GDB2ft2rVZuHBhbr755rz00kupq6vL8ccfnzvuuCNVVVXFe1x33XVp2bJlRowYkbVr12bgwIGZMWPG23qjFwAAYHe2cOTCcpfwpvbff//87Gc/y1VXXZUhQ4bklVdeSc+ePTNjxoy8//3vb/bnTZ8+PZ/85CczbNiwrFu3Lscee2x++MMfbrJk9bZw1lln5c4778zxxx+fl156KdOnT8+oUaOSJIMGDUpdXV169+6dLl26vOl9Dj/88Hz729/Of/3Xf+Wzn/1s6urq8j//8z/Fe22pysrKzJ49O9ddd12+9rWvZdy4cWnbtm169eqVT3ziE8UGzmc/+9l07tw5EydOzB//+MfsvffeOfzww3PllVeW8rch//qv/5o5c+bkYx/7WP785z9nr732Su/evXP33XfnuOOO2+L7fPOb38y///u/57DDDst+++2XCRMmZNy4cZuMu/rqqzNr1qxcdNFFqa2tzW233ZaDDz44SdK2bds88sgjufzyyzN8+PCsWrUq//Iv/5KBAwdu8glM2NXNnz8/y5cvzxFHHFE8tmHDhjzyyCOZOnVqnnrqqST/WOXqtRXsk2T58uXFBmhtbW3WrVuXFStWNFkFa/ny5W+6on1lZWUxkAoAAED59fr9onKX8La8WR9njz32yKxZs4r9rgMPPDBf+tKXMmDAgLe87+WXX576+vqcf/75adGiRf793/89J5544lZlVk4//fR86lOfyiWXXJLGxsaccsop+cxnPpPx48dv0fXz58/Pb37zm3z961/f5FxVVVWGDBmSadOm5fTTT89dd92VCy64IP/6r/+a/fffP1/60pdy0kknva163/3ud+fhhx/OVVddlfe9730pFAp517velbPPPnuL77HXXnvl2muvzeLFi9OiRYsceeSR+eEPf5g99tj0w35t27bNj3/843zyk5/MkUcembZt2+ass87KlClT3lbdza2isCUfn9zFrVy5MtXV1WloaNhpm6d9Z/Ytdwm7tR39P1gBAECSvPLKK1myZEm6d+/+piuXsGN6+eWX06VLl3zzm9/M8OHDy13OLqOioiJ33XVXzjjjjGa755v91naFHgS7r1WrVm3yyYMPf/jDOeigg3L55ZcXA6Kf+tSnctlllyVJ1q1bl86dO+faa6/NhRdemIaGhrzjHe/IrbfemhEjRiRJli1bln333Tc//OEPN/spiM3ZFX5Liw7a9p+25Y3tbP+xCAAAykE/8e3buHFjevXqlREjRuSzn/1sucthCzVHP6+sK2ABAAAAb27jxo2pr6/P5MmTU11dndNOO63cJQG7qaqqqk0+XdCuXbt07NixeHz06NGZMGFCevTokR49emTChAlp27ZtzjnnnCRJdXV1LrjggowdOzYdO3ZMhw4dMm7cuPTt2zeDBg3a7nMCAACArfHcc8/l/vvvz3HHHZfGxsZMnTo1S5YsKf45mN2HABYAAADswJ5//vl07949++67b2bMmJGWLf1RHthxXXbZZVm7dm0uuuiirFix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rSKABQAAAABskVNPPTUnn3xyevbsmZ49e+Zzn/tc9tprrzz66KMpFAq5/vrrc9VVV2X48OHp06dPZs6cmZdffjm33357kqShoSHTpk3L5MmTM2jQoBx22GG59dZbs3DhwjzwwANv+NzGxsasXLmyyQYAAACwoxDAAgAAAADetg0bNmTWrFlZs2ZN3vOe92TJkiWpr6/PkCFDimMqKytz3HHHZe7cuUmS+fPnZ/369U3GdOnSJX369CmO2ZyJEyemurq6uHXt2nXbTQwAAADgbRLAAgAAAAC22MKFC7PXXnulsrIyH/vYx3LXXXfl4IMPTn19fZKkpqamyfiampriufr6+rRu3Tr77LPPG47ZnCuuuCINDQ3FbenSpc08KwAAAIDStSx3AQAAAADAzuPAAw/MggUL8tJLL+W73/1uRo4cmYcffrh4vqKiosn4QqGwybHXe6sxlZWVqays3LrCAQAAALYRK2ABAAAAAFusdevWOeCAA9KvX79MnDgxhxxySL74xS+mtrY2STZZyWr58uXFVbFqa2uzbt26rFix4g3HAAAAAOxsBLAAAAAAgJIVCoU0Njame/fuqa2tzezZs4vn1q1bl4cffjj9+/dPkhxxxBFp1apVkzHLli3LE088URwDAAAAsLPxCUIAAAAAYItceeWVGTp0aLp27ZpVq1Zl1qxZeeihh3LfffeloqIio0ePzoQJE9KjR4/06NEjEyZMSNu2bXPOOeckSaqrq3PBBRdk7Nix6dixYzp06JBx48alb9++GTRoUJlnBwAAAFCasq6ANXHixBx55JGpqqpK586dc8YZZ+Spp55qMmbUqFGpqKhosh199NFNxjQ2NubSSy9Np06d0q5du5x22ml54YUXtudUAAAAAGCX95e//CXnnXdeDjzwwAwcODC//OUvc99992Xw4MFJkssuuyyjR4/ORRddlH79+uVPf/pT7r///lRVVRXvcd111+WMM87IiBEj8t73vjdt27bN97///bRo0aJc0wIAAADYKhWFQqFQroefdNJJ+cAHPpAjjzwyr776aq666qosXLgwv/vd79KuXbsk/whg/eUvf8n06dOL17Vu3TodOnQo7n/84x/P97///cyYMSMdO3bM2LFj8/e//z3z58/fosbNypUrU11dnYaGhrRv3775J7od9J3Zt9wl7NYWjlxY7hIAAADYCewKPQjYEewKv6VFB/Uqdwm7tV6/X1TuEgAAANgJbGkPoqyfILzvvvua7E+fPj2dO3fO/Pnzc+yxxxaPV1ZWpra2drP3aGhoyLRp03LLLbcUlym/9dZb07Vr1zzwwAM58cQTt90EAAAAAAAAAACA3VpZP0H4eg0NDUnSZHWrJHnooYfSuXPn9OzZMx/96EezfPny4rn58+dn/fr1GTJkSPFYly5d0qdPn8ydO3ezz2lsbMzKlSubbAAAAAAAAAAAAG/XDhPAKhQKGTNmTI455pj06dOneHzo0KG57bbb8uCDD2by5MmZN29eTjjhhDQ2NiZJ6uvr07p16+yzzz5N7ldTU5P6+vrNPmvixImprq4ubl27dt12EwMAAAAAAAAAAHZZZf0E4T+75JJL8tvf/jZz5sxpcvzss88u/nWfPn3Sr1+/dOvWLffee2+GDx/+hvcrFAqpqKjY7LkrrrgiY8aMKe6vXLlSCAsAAAAAAAAAAHjbdogVsC699NLcc889+elPf5p99933TcfW1dWlW7duWbx4cZKktrY269aty4oVK5qMW758eWpqajZ7j8rKyrRv377JBgAAAAAAAAAA8HaVNYBVKBRyySWX5M4778yDDz6Y7t27v+U1L774YpYuXZq6urokyRFHHJFWrVpl9uzZxTHLli3LE088kf79+2+z2gEAAAAAAAAAAMr6CcKLL744t99+e773ve+lqqoq9fX1SZLq6uq0adMmq1evzvjx43PWWWelrq4uzz77bK688sp06tQpZ555ZnHsBRdckLFjx6Zjx47p0KFDxo0bl759+2bQoEHlnB4AAAAAAAAAALCLK2sA68Ybb0ySDBgwoMnx6dOnZ9SoUWnRokUWLlyYm2++OS+99FLq6upy/PHH54477khVVVVx/HXXXZeWLVtmxIgRWbt2bQYOHJgZM2akRYsW23M6AAAAAAAAAADAbqasAaxCofCm59u0aZMf//jHb3mfPffcMzfccENuuOGG5ioNAAAAAAAAAADgLe1R7gIAAAAAAAAAAAB2VgJYAAAAAAAAAAAAJRLAAgAAAAAAAAAAKJEAFgAAAAAAAAAAQIkEsAAAAAAAAAAAAEokgAUAAAAAAAAAAFAiASwAAAAAAAAAAIASCWABAAAAAAAAAACUSAALAAAAAAAAAACgRAJYAAAAAAAAAAAAJRLAAgAAAAAAAAAAKJEAFgAAAAAAAAAAQIkEsAAAAAAAAAAAAEokgAUAAAAAAAAAAFAiASwAAAAAAAAAAIASCWABAAAAAAAAAACUSAALAAAAAAAAAACgRAJYAAAAAAAAAAAAJSopgLVkyZLmrgMAAAAA2Iaao6c3ceLEHHnkkamqqkrnzp1zxhln5KmnnmoyZtSoUamoqGiyHX300U3GNDY25tJLL02nTp3Srl27nHbaaXnhhRe2uj4AAACAcigpgHXAAQfk+OOPz6233ppXXnmluWsCAAAAAJpZc/T0Hn744Vx88cV59NFHM3v27Lz66qsZMmRI1qxZ02TcSSedlGXLlhW3H/7wh03Ojx49OnfddVdmzZqVOXPmZPXq1Rk2bFg2bNhQ8vwAAAAAyqWkANZvfvObHHbYYRk7dmxqa2tz4YUX5le/+lVz1wYAAAAANJPm6Ondd999GTVqVHr37p1DDjkk06dPz/PPP5/58+c3GVdZWZna2tri1qFDh+K5hoaGTJs2LZMnT86gQYNy2GGH5dZbb83ChQvzwAMPNMtcAQAAALankgJYffr0yZQpU/KnP/0p06dPT319fY455pj07t07U6ZMyV//+tfmrhMAAAAA2ArboqfX0NCQJE0CVkny0EMPpXPnzunZs2c++tGPZvny5cVz8+fPz/r16zNkyJDisS5duqRPnz6ZO3fuZp/T2NiYlStXNtkAAAAAdhQlBbBe07Jly5x55pn59re/nWuvvTbPPPNMxo0bl3333Tfnn39+li1b1lx1AgAAAADNoLl6eoVCIWPGjMkxxxyTPn36FI8PHTo0t912Wx588MFMnjw58+bNywknnJDGxsYkSX19fVq3bp199tmnyf1qampSX1+/2WdNnDgx1dXVxa1r164lzh4AAACg+W1VAOuxxx7LRRddlLq6ukyZMiXjxo3LM888kwcffDB/+tOfcvrppzdXnQAAAABAM2iunt4ll1yS3/72t/nWt77V5PjZZ5+dU045JX369Mmpp56aH/3oR3n66adz7733vun9CoVCKioqNnvuiiuuSENDQ3FbunTplk0WAAAAYDtoWcpFU6ZMyfTp0/PUU0/l5JNPzs0335yTTz45e+zxjzxX9+7d87WvfS0HHXRQsxYLAAAAAJSmOXt6l156ae6555488sgj2Xfffd90bF1dXbp165bFixcnSWpra7Nu3bqsWLGiySpYy5cvT//+/Td7j8rKylRWVm7pVAEAAAC2q5JWwLrxxhtzzjnn5Pnnn8/dd9+dYcOGFRs1r9lvv/0ybdq0ZikSAAAAANg6zdHTKxQKueSSS3LnnXfmwQcfTPfu3d/yuS+++GKWLl2aurq6JMkRRxyRVq1aZfbs2cUxy5YtyxNPPPGGASwAAACAHVlJK2C99rbam2ndunVGjhxZyu0BAAAAgGbWHD29iy++OLfffnu+973vpaqqKvX19UmS6urqtGnTJqtXr8748eNz1llnpa6uLs8++2yuvPLKdOrUKWeeeWZx7AUXXJCxY8emY8eO6dChQ8aNG5e+fftm0KBBzTNZAAAAgO2opADW9OnTs9dee+X9739/k+Pf+c538vLLLwteAQAAAMAOpjl6ejfeeGOSZMCAAZvce9SoUWnRokUWLlyYm2++OS+99FLq6upy/PHH54477khVVVVx/HXXXZeWLVtmxIgRWbt2bQYOHJgZM2akRYsWWz9RAAAAgO2spE8Qfv7zn0+nTp02Od65c+dMmDBhq4sCAAAAAJpXc/T0CoXCZrdRo0YlSdq0aZMf//jHWb58edatW5fnnnsuM2bMSNeuXZvcZ88998wNN9yQF198MS+//HK+//3vbzIGAAAAYGdRUgDrueeeS/fu3Tc53q1btzz//PNbXRQAAAAA0Lz09AAAAAC2jZICWJ07d85vf/vbTY7/5je/SceOHbe6KAAAAACgeenpAQAAAGwbJQWwPvCBD+QTn/hEfvrTn2bDhg3ZsGFDHnzwwXzyk5/MBz7wgeauEQAAAADYSnp6AAAAANtGy1Iuuuaaa/Lcc89l4MCBadnyH7fYuHFjzj///EyYMKFZCwQAAAAAtp6eHgAAAMC2UdIKWK1bt84dd9yR3//+97ntttty55135plnnsk3v/nNtG7deovvM3HixBx55JGpqqpK586dc8YZZ+Spp55qMqZQKGT8+PHp0qVL2rRpkwEDBuTJJ59sMqaxsTGXXnppOnXqlHbt2uW0007LCy+8UMrUAAAAAGCX1Fw9PQAAAACaKimA9ZqePXvm/e9/f4YNG5Zu3bq97esffvjhXHzxxXn00Ucze/bsvPrqqxkyZEjWrFlTHDNp0qRMmTIlU6dOzbx581JbW5vBgwdn1apVxTGjR4/OXXfdlVmzZmXOnDlZvXp1hg0blg0bNmzN9AAAAABgl7O1PT0AAAAAmirpE4QbNmzIjBkz8pOf/CTLly/Pxo0bm5x/8MEHt+g+9913X5P96dOnp3Pnzpk/f36OPfbYFAqFXH/99bnqqqsyfPjwJMnMmTNTU1OT22+/PRdeeGEaGhoybdq03HLLLRk0aFCS5NZbb03Xrl3zwAMP5MQTTyxligAAAACwS2munh4AAAAATZUUwPrkJz+ZGTNm5JRTTkmfPn1SUVHRLMU0NDQkSTp06JAkWbJkSerr6zNkyJDimMrKyhx33HGZO3duLrzwwsyfPz/r169vMqZLly7p06dP5s6du9kAVmNjYxobG4v7K1eubJb6AQAAAGBHta16egAAAAC7u5ICWLNmzcq3v/3tnHzyyc1WSKFQyJgxY3LMMcekT58+SZL6+vokSU1NTZOxNTU1ee6554pjWrdunX322WeTMa9d/3oTJ07M1Vdf3Wy1AwAAAMCOblv09AAAAABI9ijlotatW+eAAw5o1kIuueSS/Pa3v823vvWtTc69/m28QqHwlm/ovdmYK664Ig0NDcVt6dKlpRcOAAAAADuBbdHTAwAAAKDEANbYsWPzxS9+MYVCoVmKuPTSS3PPPffkpz/9afbdd9/i8dra2iTZZCWr5cuXF1fFqq2tzbp167JixYo3HPN6lZWVad++fZMNAAAAAHZlzd3TAwAAAOAfSvoE4Zw5c/LTn/40P/rRj9K7d++0atWqyfk777xzi+5TKBRy6aWX5q677spDDz2U7t27NznfvXv31NbWZvbs2TnssMOSJOvWrcvDDz+ca6+9NklyxBFHpFWrVpk9e3ZGjBiRJFm2bFmeeOKJTJo0qZTpAQAAAMAup7l6egAAAAA0VVIAa++9986ZZ5651Q+/+OKLc/vtt+d73/teqqqqiitdVVdXp02bNqmoqMjo0aMzYcKE9OjRIz169MiECRPStm3bnHPOOcWxF1xwQcaOHZuOHTumQ4cOGTduXPr27ZtBgwZtdY0AAAAAsCtorp4eAAAAAE2VFMCaPn16szz8xhtvTJIMGDBgk/uPGjUqSXLZZZdl7dq1ueiii7JixYocddRRuf/++1NVVVUcf91116Vly5YZMWJE1q5dm4EDB2bGjBlp0aJFs9QJAAAAADu75urpAQAAANBURaFQKJRy4auvvpqHHnoozzzzTM4555xUVVXlz3/+c9q3b5+99tqruevcplauXJnq6uo0NDSkffv25S6nJH1n9i13Cbu1hSMXlrsEAAAAdgK7Qg+Cnduu0tPbFX5Liw7qVe4Sdmu9fr+o3CUAAACwE9jSHkRJK2A999xzOemkk/L888+nsbExgwcPTlVVVSZNmpRXXnklX/3qV0suHAAAAABofnp6AAAAANvGHqVc9MlPfjL9+vXLihUr0qZNm+LxM888Mz/5yU+arTgAAAAAoHno6QEAAABsGyWtgDVnzpz8/Oc/T+vWrZsc79atW/70pz81S2EAAAAAQPPR0wMAAADYNkpaAWvjxo3ZsGHDJsdfeOGFVFVVbXVRAAAAAEDz0tMDAAAA2DZKCmANHjw4119/fXG/oqIiq1evzn//93/n5JNPbq7aAAAAAIBmoqcHAAAAsG2U9AnC6667Lscff3wOPvjgvPLKKznnnHOyePHidOrUKd/61reau0YAAAAAYCvp6QEAAABsGyUFsLp06ZIFCxbkW9/6Vn79619n48aNueCCC/KhD30obdq0ae4aAQAAAICtpKcHAAAAsG2UFMBKkjZt2uQjH/lIPvKRjzRnPQAAAADANqKnBwAAAND8Sgpg3XzzzW96/vzzzy+pGAAAAABg29DTAwAAANg2SgpgffKTn2yyv379+rz88stp3bp12rZtq1kDAAAAADsYPT0AAACAbWOPUi5asWJFk2316tV56qmncswxx+Rb3/pWc9cIAAAAAGwlPT0AAACAbaOkANbm9OjRI5///Oc3eZMOAAAAANgx6ekBAAAAbL1mC2AlSYsWLfLnP/+5OW8JAAAAAGxDenoAAAAAW6dlKRfdc889TfYLhUKWLVuWqVOn5r3vfW+zFAYAAAAANJ/m6OlNnDgxd955Z37/+9+nTZs26d+/f6699toceOCBTe579dVX56abbsqKFSty1FFH5ctf/nJ69+5dHNPY2Jhx48blW9/6VtauXZuBAwfmK1/5Svbdd9/mmSwAAADAdlRSAOuMM85osl9RUZF3vOMdOeGEEzJ58uTmqAsAAAAAaEbN0dN7+OGHc/HFF+fII4/Mq6++mquuuipDhgzJ7373u7Rr1y5JMmnSpEyZMiUzZsxIz549c80112Tw4MF56qmnUlVVlSQZPXp0vv/972fWrFnp2LFjxo4dm2HDhmX+/Plp0aJFs84bAAAAYFsrKYC1cePG5q4DAAAAANiGmqOnd9999zXZnz59ejp37pz58+fn2GOPTaFQyPXXX5+rrroqw4cPT5LMnDkzNTU1uf3223PhhRemoaEh06ZNyy233JJBgwYlSW699dZ07do1DzzwQE488cStrhMAAABge9qj3AUAAAAAADunhoaGJEmHDh2SJEuWLEl9fX2GDBlSHFNZWZnjjjsuc+fOTZLMnz8/69evbzKmS5cu6dOnT3HM6zU2NmblypVNNgAAAIAdRUkrYI0ZM2aLx06ZMqWURwAAAAAAzai5e3qFQiFjxozJMccckz59+iRJ6uvrkyQ1NTVNxtbU1OS5554rjmndunX22WefTca8dv3rTZw4MVdfffUW1w8AAACwPZUUwHr88cfz61//Oq+++moOPPDAJMnTTz+dFi1a5PDDDy+Oq6ioaJ4qAQAAAICt0tw9vUsuuSS//e1vM2fOnE3Ovf4ehULhLe/7ZmOuuOKKJgGylStXpmvXrltUJwAAAMC2VlIA69RTT01VVVVmzpxZfFNtxYoV+fCHP5z3ve99GTt2bLMWCQAAAABsnebs6V166aW555578sgjj2TfffctHq+trU3yj1Wu6urqiseXL19eXBWrtrY269aty4oVK5qsgrV8+fL0799/s8+rrKxMZWXllk8WAAAAYDvao5SLJk+enIkTJzZpkOyzzz655pprMnny5GYrDgAAAABoHs3R0ysUCrnkkkty55135sEHH0z37t2bnO/evXtqa2sze/bs4rF169bl4YcfLoarjjjiiLRq1arJmGXLluWJJ554wwAWAAAAwI6spBWwVq5cmb/85S/p3bt3k+PLly/PqlWrmqUwAAAAAKD5NEdP7+KLL87tt9+e733ve6mqqkp9fX2SpLq6Om3atElFRUVGjx6dCRMmpEePHunRo0cmTJiQtm3b5pxzzimOveCCCzJ27Nh07NgxHTp0yLhx49K3b98MGjSoeScNAAAAsB2UFMA688wz8+EPfziTJ0/O0UcfnSR59NFH8x//8R8ZPnx4sxYIAAAAAGy95ujp3XjjjUmSAQMGNDk+ffr0jBo1Kkly2WWXZe3atbnooouyYsWKHHXUUbn//vtTVVVVHH/dddelZcuWGTFiRNauXZuBAwdmxowZadGixdZPFAAAAGA7qygUCoW3e9HLL7+ccePG5Zvf/GbWr1+fJGnZsmUuuOCCfOELX0i7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YAEAAAAAAAAAAJRIAAsAAAAAAAAAAKBEAlgAAAAAAAAAAAAlEsCCbWzixIk58sgjU1VVlc6dO+eMM87IU089Ve6yYJcxceLEVFRUZPTo0eUuBXZqf/rTn3LuueemY8eOadu2bQ499NDMnz+/3GXBTuvVV1/Nf/7nf6Z79+5p06ZN3vnOd+Z//ud/snHjxnKXBjuFRx55JKeeemq6dOmSioqK3H333U3OFwqFjB8/Pl26dEmbNm0yYMCAPPnkk+UpFoBdjn4ebHt6erD19POgeennwdbb3Xt6AliwjT388MO5+OKL8+ijj2b27Nl59dVXM2TIkKxZs6bcpcFOb968ebnpppvy7ne/u9ylwE5txYoVee9735tWrVrlRz/6UX73u99l8uTJ2XvvvctdGuy0rr322nz1q1/N1KlTs2jRokyaNClf+MIXcsMNN5S7NNgprFmzJoccckimTp262fOTJk3KlClTMnXq1MybNy+1tbUZPHhwVq1atZ0rBWBXpJ8H25aeHmw9/Txofvp5sPV2955eRaFQKJS7CNid/PWvf03nzp3z8MMP59hjjy13ObDTWr16dQ4//PB85StfyTXXXJNDDz00119/fbnLgp3Spz/96fz85z/Pz372s3KXAruMYcOGpaamJtOmTSseO+uss9K2bdvccsstZawMdj4VFRW56667csYZZyT5x5tyXbp0yejRo3P55ZcnSRobG1NTU5Nrr702F154YRmrBWBXpJ8HzUdPD5qHfh40P/08aF67Y0/PCliwnTU0NCRJOnToUOZKYOd28cUX55RTTsmgQYPKXQrs9O65557069cv73//+9O5c+ccdthh+frXv17usmCndswxx+QnP/lJnn766STJb37zm8yZMycnn3xymSuDnd+SJUtSX1+fIUOGFI9VVlbmuOOOy9y5c8tYGQC7Kv08aD56etA89POg+ennwba1O/T0Wpa7ANidFAqFjBkzJsccc0z69OlT7nJgpzVr1qz8+te/zrx588pdCuwS/vjHP+bGG2/MmDFjcuWVV+ZXv/pVPvGJT6SysjLnn39+ucuDndLll1+ehoaGHHTQQWnRokU2bNiQz33uc/ngBz9Y7tJgp1dfX58kqampaXK8pqYmzz33XDlKAmAXpp8HzUdPD5qPfh40P/082LZ2h56eABZsR5dcckl++9vfZs6cOeUuBXZaS5cuzSc/+cncf//92XPPPctdDuwSNm7cmH79+mXChAlJksMOOyxPPvlkbrzxRg0bKNEdd9yRW2+9Nbfffnt69+6dBQsWZPTo0enSpUtGjhxZ7vJgl1BRUdFkv1AobHIMALaWfh40Dz09aF76edD89PNg+9iVe3oCWLCdXHrppbnnnnvyyCOPZN999y13ObDTmj9/fpYvX54jjjiieGzDhg155JFHMnXq1DQ2NqZFixZlrBB2PnV1dTn44IObHOvVq1e++93vlqki2Pn9x3/8Rz796U/nAx/4QJKkb9++ee655zJx4kQNG9hKtbW1Sf7x1lxdXV3x+PLlyzd5gw4AtoZ+HjQfPT1oXvp50Pz082Db2h16enuUuwDY1RUKhVxyySW588478+CDD6Z79+7lLgl2agMHDszChQuzYMGC4tavX7986EMfyoIFCzRqoATvfe9789RTTzU59vTTT6dbt25lqgh2fi+//HL22KPpH7datGiRjRs3lqki2HV07949tbW1mT17dvHYunXr8vDDD6d///5lrAyAXYV+HjQ/PT1oXvp50Pz082Db2h16elbAgm3s4osvzu23357vfe97qaqqKn7btLq6Om3atClzdbDzqaqqSp8+fZoca9euXTp27LjJcWDLfOpTn0r//v0zYcKEjBgxIr/61a9y00035aabbip3abDTOvXUU/O5z30u++23X3r37p3HH388U6ZMyUc+8pFylwY7hdWrV+cPf/hDcX/JkiVZsGBBOnTokP322y+jR4/OhAkT0qNHj/To0SMTJkxI27Ztc84555SxagB2Ffp50Pz09KB56edB89PPg623u/f0KgqFQqHcRcCu7I2+Vzp9+vSMGjVq+xYDu6gBAwbk0EMPzfXXX1/uUmCn9YMf/CBXXHFFFi9enO7du2fMmDH56Ec/Wu6yYKe1atWqfOYzn8ldd92V5cuXp0uXLvngBz+Y//qv/0rr1q3LXR7s8B566KEcf/zxmxwfOXJkZsyYkUKhkKuvvjpf+9rXsmLFihx11FH58pe/7D/eAdAs9PNg+9DTg62jnwfNSz8Ptt7u3tMTwAIAAAAAAAAAACjRHm89BAAAAAAAAAAAgM0RwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAA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s5Lk1ltvzec+97k899xzOeOMM/JP//RP+dznPpfnn39+l/XcdtttmTFjRtasWZO3vOUt+eQnP5lLL730NddQXV2d2traXZ6vqqrKl7/85dxzzz15+OGHU1tbmxkzZuT9739/khTX84tf/CKHHXZYTjvttCQphhvOP//84htiL7/8cq644op87WtfS/v27XPJJZdk6tSpxWetX78+//Iv/5K77747L774YgYPHpwbbrgh73jHO3Za2wUXXJDnn3++uAX3/fffn2uvvTarV69OmzZtctJJJ+Wmm27K3//937/m34NXFAqFXHDBBenfv39+8pOf5JBD/r/3v4499th86lOfKn7+wx/+kIkTJ+aBBx7IIYcckpNPPjk33XRT3vKWt+zWs17te9/7Xi699NLi39ckO6x76NChOfbYY3PjjTcWdxp7tb/9+/29730vU6dOzWOPPZa6urqcf/75ufrqqysSloN9wWWXXZb3vOc9GT58eK699tri+Jo1a9LY2JiRI0cWx6qrq3Pqqadm6dKlufjii7NixYps27atxZy6urrU19dn6dKlOeOMM3b6zObm5jQ3Nxc/b9iwoRVWtnft7TepaemNvlkOAACwL/rbnmRtbW2uvPLKnHLKKfnzn/+cN73pTUmSK6+8MgsXLsyzzz6b2trafOhDH8q//du/FXeInzp1au6+++588pOfzLXXXpunnnoq27dvz4YNG95Qj/EV48ePz8yZMzN58uQMHDhwp3Pe8pa3ZMKECZkwYUJx7Nhjj83ZZ59d7HM+//zzueKKK/Ld734369evz5FHHpnrr79+lwGz1+vh/W1vuTXYAQsAAICKO/TQQ3PooYfm7rvvbvEf2HfH1VdfncmTJ2flypUZMGBAPvjBD+all15Kkvz0pz/NJZdckk996lNZuXJlRowYkeuuu+4173frrbfm6quvznXXXZdf//rXmTZtWj772c/u9CcG36jPfvazxZ2+PvzhD+eDH/xgfv3rX+8wr0+fPvn2t7+dJHniiSfS0NCQm266qXj+9ttvT+fOnfNf//VfmTFjRj73uc9l0aJFSf4afnrPe96TxsbG3HvvvVmxYkXe+c53ZtiwYS12E3stmzdvzsSJE7Ns2bL86Ec/yiGHHJJzzjknL7/88m5dv3Llyjz22GOZNGlSi/DVK17ZbeuFF17IaaedlkMPPTSPPPJIlixZkkMPPTRnnnlmtm7dulvPerXa2to89NBD+fOf/7xb84cMGdJi17KHHnooHTp0yCmnnJIk+eEPf5gPf/jD+eQnP5nHH388t9xyS+bNm/e6/xzBgWrBggX57//+70yfPn2Hc42NjUmSXr16tRjv1atX8VxjY2Pat2+/w5b/fztnZ6ZPn56ampri0adPnz1dCgAAABxQNm3alG984xs58sgj07179+J4ly5dMm/evDz++OO56aabcuutt+aGG25oce1vf/vbfPOb38y3v/3trFy5MklK7jEOGTIko0ePzpQpU0pey8svv5xRo0Zl6dKlmT9/fh5//PFcf/31adOmzU7n7ws9PAEsAAAAKq5t27aZN29ebr/99hx22GF517velauuuiq/+tWvXvfayZMn5z3veU8GDBiQa665Jk8//XR++9vfJklmz56dUaNGZfLkyRkwYEAuvfTSjBo16jXv9/nPfz4zZ87M2LFj069fv4wdOzaf/vSnc8stt7zmdd///veLQbJXjs9//vMt5rz//e/PJz7xiQwYMCCf//znM3jw4MyePXuHe7Vp0ybdunVLkvTs2TO1tbWpqakpnj/mmGPy7//+7+nfv38++tGPZvDgwfnRj36UJPnxj3+cVatW5Vvf+lYGDx6c/v375//+3/+bww47LP/v//2/1/37mfz15yDHjh2b/v3759hjj83cuXOzatWqPP7447t1/ZNPPpkkOeqoo15z3oIFC3LIIYfka1/7WgYOHJh/+Id/yG233ZZnnnkmDz/88G4969VmzZqVP//5z6mtrc0xxxyTSy65JPfdd98u57dv3z61tbWpra1Nu3btctFFF+XjH/94Pv7xjydJrrvuunzmM5/J+eefn7e+9a0ZMWJEPv/5z7/uPw9wIFq7dm0+9alPZf78+enQocMu51VVVbX4XCgUdhh7tdebM2XKlKxfv754rF279o0VDwAAAAegv+1JdunSJffcc0/uuuuuFi9F/uu//muGDBmSt7zlLRkzZkwmTZqUb37zmy3us3Xr1txxxx057rjjcswxx+xxj3H69Om5//7785Of/KSkdT344IP5+c9/nu985zsZMWJE3vrWt2b06NG77O3uCz08e+UDAACwT3jf+96X97znPfnJT36Sn/3sZ7n//vszY8aMfO1rX8sFF1ywy+uOOeaY4l/37t07SdLU1JSjjjoqTzzxRM4555wW8//xH/8x3//+zn926M9//nPWrl2bCy+8MBdddFFx/KWXXmoRgNqZ0047LTfffHOLsVdCVK/4259GfOXzK2+UvRF/u+bkr+tuampKkqxYsSKbNm1q8ZZbkmzZsiW/+93vduv+v/vd7/LZz342jz76aP7yl78Ud7565plnUl9f/7rXFwqFJDuGMF5txYoV+e1vf5suXbq0GH/xxRd3u9ZXO/roo7N69eqsWLEiS5YsySOPPJIxY8bkggsuyNe+9rVdXrdt27a8733vy5vf/OYWu42tWLEiy5Yta/G23Pbt2/Piiy/mhRdeSKdOnUqqE/ZHK1asSFNTUwYNGlQc2759ex555JHMmTMnTzzxRJK/7nL1yr+Pk7/+O/mVXbFqa2uzdevWrFu3rsUuWE1NTRkyZMgun11dXZ3q6upyLwkAAAD2a3/bk/zf//3ffPnLX86oUaPy85//PH379k2S/L//9/9y44035re//W02bdqUl156KV27dm1xn759+xZ/sjDZ8x7j0UcfnY9+9KO58sors3Tp0je8rpUrV+aII47IgAEDdmv+vtDDE8ACAABgn9GhQ4eMGDEiI0aMyL/927/lE5/4RP793//9NQNY7dq1K/71K4GfVwJDO9tR5ZVw0M68ct2tt96aE044ocW5XW1v/YrOnTvnyCOPfM05O/N6IaWd+ds1v3KPV2p/+eWX07t3753uIPXKT/+9njFjxqRPnz659dZbU1dXl5dffjn19fW7/bOArzRGfv3rX+fYY4/d5byXX345gwYNyje+8Y0dzv1tw+eNOuSQQ3L88cfn+OOPz6c//enMnz8/H/nIR3L11VenX79+O73mn//5n/PMM89k2bJladv2/2uXvPzyy7nmmmsyduzYHa55rR2A4EA0bNiwrFq1qsXYxz72sRx11FG58sor89a3vjW1tbVZtGhRjjvuuCR/fYN28eLF+cIXvpAkGTRoUNq1a5dFixZl3LhxSZKGhoasXr06M2bM2LsLAgAAgP3cq3uSgwYNSk1NTW699dZce+21efTRR/OBD3wg11xzTc4444zU1NRkwYIFmTlz5g73+Vvl6DFec801GTBgQO6+++4dzh1yyCE79Gm3bdtW/OuOHTvu1jNesS/08ASwAAAA2GcdffTRO/0D+u466qij8vOf/7zF2PLly3c5v1evXvm7v/u7/P73v8+HPvShkp+7K48++mg++tGPtvj8Skjh1dq3b5/kr29qvRHvfOc709jYmLZt2+Ytb3nLG67xueeey69//evccsst+ad/+qckyZIlS97QPY499tgcffTRmTlzZs4999wWW54nyfPPP5/DDjss73znO3PXXXelZ8+eO7x1V05HH310kmTz5s07PT9r1qzcdddd+dnPfrbDW33vfOc788QTT5QUroMDTZcuXXbYBa9z587p3r17cXzChAmZNm1a+vfvn/79+2fatGnp1KlTzjvvvCRJTU1NLrzwwkyaNCndu3dPt27dMnny5AwcODDDhw/f62sCAACAA0lVVVUOOeSQbNmyJUny05/+NH379s3VV19dnPP000+/7n32tMeYJH369Mnll1+eq666Kn//93/f4tyb3vSmNDQ0FD9v2LAha9asKX4+5phj8uyzz+Y3v/nNbu2CtS/08ASwAAAAqLjnnnsu73//+/Pxj388xxxzTLp06ZLly5dnxowZee9731vyfcePH59TTjkls2bNypgxY/LQQw/lvvvue81dp6ZOnZpPfvKT6dq1a0aNGpXm5uYsX74869aty8SJE3d5XXNzcxobG1uMtW3bNj169Ch+/ta3vpXBgwfn5JNPzje+8Y38/Oc/z9y5c3d6v759+6aqqirf//738+53vzsdO3bMoYce+rprHj58eE466aScffbZ+cIXvpC3ve1t+eMf/5h77703Z599dgYPHvya1x9++OHp3r17vvrVr6Z379555pln8pnPfOZ1n/u3qqqqctttt2X48OE55ZRTctVVV+Woo47Kpk2b8r3vfS8PPPBAFi9enA996EP54he/mPe+97353Oc+lyOOOCLPPPNMvvOd7+Rf/uVfcsQRR7yh5ybJ//k//yfvete7MmTIkNTW1mbNmjWZMmVKBgwYkKOOOmqH+Q8++GCuuOKKfOlLX0qPHj2K/xt27NgxNTU1+bd/+7eMHj06ffr0yfvf//4ccsgh+dWvfpVVq1bl2muvfcP1wYHuiiuuyJYtW3LppZdm3bp1OeGEE/LAAw+0+KnRG264IW3bts24ceOyZcuWDBs2LPPmzXvdnQYBAACAlv62J7lu3brMmTMnmzZtypgxY5IkRx55ZJ555pksWLAgxx9/fH7wgx9k4cKFr3vfPe0xvmLKlCm59dZbs2bNmpx77rnF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V1enurq6zCsBAAAAAAAAAAAONhX/CcJXKxQKxZ8YfLXnnnsua9euTe/evZMkgwYNSrt27bJo0aLinIaGhqxevfo1A1gAAAAAAAAAAADlUNEdsK666qqMGjUqffr0ycaNG7NgwYI8/PDDuf/++7Np06ZMnTo173vf+9K7d+889dRTueqqq9KjR4+cc845SZKamppceOGFmTRpUrp3755u3bpl8uTJGThwYIYPH17JpQEAAAAAAAAAAAeBigaw/vSnP+UjH/lIGhoaUlNTk2OOOSb3339/RowYkS1btmTVqlX5+te/nueffz69e/fOaaedlrvuuitdunQp3uOGG25I27ZtM27cuGzZsiXDhg3LvHnz0qZNmwquDAAAAAAAAAAAOBhUNIA1d+7cXZ7r2LFjfvjDH77uPTp06JDZs2dn9uzZ5SwNAAAAAAAAAADgdR1S6QIAAAAAAAAAAAD2VwJYAAAAAMBuufnmm3PMMceka9eu6dq1a0466aTcd999xfMXXHBBqqqqWhwnnnhii3s0Nzdn/Pjx6dGjRzp37pyzzjorzz777N5eCgAAAEDZCGABAAAAALvliCOOyPXXX5/ly5dn+fLlOf300/Pe9743jz32WHHOmWeemYaGhuJx7733trjHhAkTsnDhwixYsCBLlizJpk2bMnr06Gzfvn1vLwcAAACgLNpWugAAAAAAYP8wZsyYFp+vu+663HzzzXn00Ufz9re/PUlSXV2d2tranV6/fv36zJ07N3fccUeGDx+eJJk/f3769OmTBx98MGeccUbrLgAAAACgFdgBCwAAAAB4w7Zv354FCxZk8+bNOemkk4rjDz/8cHr27JkBAwbkoosuSlNTU/HcihUrsm3btowcObI4VldXl/r6+ixdunSXz2pubs6GDRtaHAAAAAD7CgEsAAAAAGC3rVq1Koceemiqq6tzySWXZOHChTn66KOTJKNGjco3vvGNPPTQQ5k5c2aWLVuW008/Pc3NzUmSxsbGtG/fPocffniLe/bq1SuNjY27fOb06dNTU1NTPPr06dN6CwQAAAB4g/wEIQAAAACw2972trdl5cqVef755/Ptb387559/fhYvXpyjjz465557bnFefX19Bg8enL59++YHP/hBxo4du8t7FgqFVFVV7fL8lClTMnHixOLnDRs2CGEBAAAA+wwBLAAAAABgt7Vv3z5HHnlkkmTw4MFZtmxZbrrpptxyyy07zO3du3f69u2bJ598MklSW1ubrVu3Zt26dS12wWpqasqQIUN2+czq6upUV1eXeSUAAAAA5eEnCAEAAACAkhUKheJPDL7ac889l7Vr16Z3795JkkGDBqVdu3ZZtGhRcU5DQ0NWr179mgEsAAAAgH2ZHbAAAAAAgN1y1VVXZdSoUenTp082btyYBQsW5OGHH87999+fTZs2ZerUqXnf+96X3r1756mnnspVV12VHj165JxzzkmS1NTU5MILL8ykSZPSvXv3dOvWLZMnT87AgQMzfPjwCq8OAAAAoDQCWAAAAADAbvnTn/6Uj3zkI2loaEhNTU2OOeaY3H///RkxYkS2bNmSVatW5etf/3qef/759O7dO6eddlruuuuudOnSpXiPG264IW3bts24ceOyZcuWDBs2LPPmzUubNm0quDIAAACA0glgAQAAAAC7Ze7cubs817Fjx/zwhz983Xt06NAhs2fPzuzZs8tZGgAAAEDFHFLpAgAAAAAAAAAAAPZXAlgAAAAAAAAAAAAlEsACAAAAAAAAAAAokQAWAAAAAAAAAABAiQSwAAAAAAAAAAAASiSABQAAAAAAAAAAUCIBLAAAAAAAAAAAgBIJYAEAAAAAAAAAAJRIAAsAAAAAAAAAAKBEAlgAAAAAAAAAAAAlEsACAAAAAAAAAAAokQAWAAAAAAAAAABAiQSwAAAAAAAAAAAASiSABQAAAAAAAAAAUCIBLAAAAAAAAAAAgBIJYAEAAAAAAAAAAJRIAAsAAAAAAAAAAKBEAlgAAAAAAAAAAAAlqmgA6+abb84xxxyTrl27pmvXrjnppJNy3333Fc8XCoVMnTo1dXV16dixY4YOHZrHHnusxT2am5szfvz49OjRI507d85ZZ52VZ599dm8vBQAAAAAAAAAAOAhVNIB1xBFH5Prrr8/y5cuzfPnynH766Xnve99bDFnNmDEjs2bNypw5c7Js2bLU1tZmxIgR2bhxY/EeEyZMyMKFC7NgwYIsWbIkmzZtyujRo7N9+/ZKLQsAAAAAAAAAADhIVDSANWbMmLz73e/OgAEDMmDAgFx33XU59NBD8+ijj6ZQKOTGG2/M1VdfnbFjx6a+vj633357Xnjhhdx5551JkvXr12fu3LmZOXNmhg8fnuOOOy7z58/PqlWr8uCDD1ZyaQAAAAAAAAAAwEGgogGsv7V9+/YsWLAgmzdvzkknnZQ1a9aksbExI0eOLM6prq7OqaeemqVLlyZJVqxYkW3btrWYU1dXl/r6+uKcnWlubs6GDRtaHAAAAAAAAAAAAG9UxQNYq1atyqGHHprq6upccsklWbhwYY4++ug0NjYmSXr16tVifq9evYrnGhsb0759+xx++OG7nLMz06dPT01NTfHo06dPmVcFAAAAAAAAAAAcDCoewHrb296WlStX5tFHH80///M/5/zzz8/jjz9ePF9VVdVifqFQ2GHs1V5vzpQpU7J+/frisXbt2j1bBAAAAAAAAAAAcFCqeACrffv2OfLIIzN48OBMnz4973jHO3LTTTeltrY2SXbYyaqpqam4K1ZtbW22bt2adevW7XLOzlRXV6dr164tDgAAAAAAAAAAgDeq4gGsVysUCmlubk6/fv1SW1ubRYsWFc9t3bo1ixcvzpAhQ5IkgwYNSrt27VrMaWhoyOrVq4tzAAAAAAAAAAAAWkvbSj78qquuyqhRo9KnT59s3LgxCxYsyMMPP5z7778/VVVVmTBhQqZNm5b+/funf//+mTZtWjp16pTzzjsvSVJTU5MLL7wwkyZNSvfu3dOtW7dMnjw5AwcOzPDhwyu5NAAAAAAAAAAA4CBQ0QDWn/70p3zkIx9JQ0NDampqcswxx+T+++/PiBEjkiRXXHFFtmzZkksvvTTr1q3LCSeckAceeCBdunQp3uOGG25I27ZtM27cuGzZsiXDhg3LvHnz0qZNm0otCwAAAAAAAAAAOEhU9CcI586dm6eeeirNzc1pamrKgw8+WAxfJUlVVVWmTp2ahoaGvPjii1m8eHHq6+tb3KNDhw6ZPXt2nnvuubzwwgv53ve+lz59+uztpQAAAADAAe/mm2/OMccck65du6Zr16456aSTct999xXPFwqFTJ06NXV1denYsWOGDh2axx57rMU9mpubM378+PTo0SOdO3fOWWedlWeffXZvLwUAAACgbCoawAIAAAAA9h9HHHFErr/++ixfvjzLly/P6aefnve+973FkNWMGTMya9aszJkzJ8uWLUttbW1GjBiRjRs3Fu8xYcKELFy4MAsWLMiSJUuyadOmjB49Otu3b6/UsgAAAAD2iAAWAAAAALBbxowZk3e/+90ZMGBABgwYkOuuuy6HHnpoHn300RQKhdx44425+uqr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adOmTVatWpX3vve9GTBgQM4///wMGDAgP/vZz9KlS5fifW644YacffbZGTduXN71rnelU6dO+d73vpc2bdpUamkAAAAAcMBZvHhxLrvssjz66KNZtGhRXnrppYwcOTKbN29uMe/MM89MQ0ND8bj33ntbnJ8wYUIWLlyYBQsWZMmSJdm0aVNGjx6d7du3783lAAAAAJRF21IuWrNmzW79XODrKRQKr3m+Y8eO+eEPf/i69+nQoUNmz56d2bNn73FNAAAAAHAgKkdP7/7772/x+bbbbkvPnj2zYsWKnHLKKcXx6urq1NbW7vQe69evz9y5c3PHHXcUd7CfP39++vTpkwcffDBnnHHGHtUIAAAAsLeVtAPWkUcemdNOOy3z58/Piy++WO6aAAAAAIAya42e3vr165Mk3bp1azH+8MMPp2fPnhkwYEAuuuiiNDU1Fc+tWLEi27Zty8iRI4tjdXV1qa+vz9KlS3f6nObm5mzYsKHFAQAAALCvKCmA9ctf/jLHHXdcJk2alNra2lx88cX5+c9/Xu7aAAAAAIAyKXdPr1AoZOLEiTn55JNTX19fHB81alS+8Y1v5KGHHsrMmTOzbNmynH766Wlubk6SNDY2pn379jn88MNb3K9Xr15pbGzc6bOmT5+empqa4tGnT5+S6wYAAAAot5ICWPX19Zk1a1b+8Ic/5LbbbktjY2NOPvnkvP3tb8+sWbPy5z//udx1AgAAAAB7oNw9vcsvvzy/+tWv8p//+Z8txs8999y85z3vSX19fcaMGZP77rsvv/nNb/KDH/zgNe9XKBRSVVW103NTpkzJ+vXri8fatWvfUK0AAAAAramkANYr2rZtm3POOSff/OY384UvfCG/+93vMnny5BxxxBH56Ec/moaGhnLVCQAAAACUQTl6euPHj88999yTH//4xzniiCNec27v3r3Tt2/fPPnkk0mS2trabN26NevWrWsxr6mpKb169drpPaqrq9O1a9cWBwAAAMC+Yo8CWMuXL8+ll16a3r17Z9asWZk8eXJ+97vf5aGHHsof/vCHvPe97y1XnQAAAABAGexJT69QKOTyyy/Pd77znTz00EPp16/f6z7vueeey9q1a9O7d+8kyaBBg9KuXbssWrSoOKehoSGrV6/OkCFD9nyBAAAAAHtZ21IumjVrVm677bY88cQTefe7352vf/3refe7351DDvlrnqtfv3655ZZbctRRR5W1WAAAAACgNOXo6V122WW58847893vfjddunRJY2NjkqSmpiYdO3bMpk2bMnXq1Lzvfe9L796989RTT+Wqq65Kjx49cs455xTnXnjhhZk0aVK6d++ebt26ZfLkyRk4cGCGDx/e+n8jAAAAAMqspADWzTffnI9//OP52Mc+ltra2p3OefOb35y5c+fuUXEAAAAAQHmUo6d38803J0mGDh3aYvy2227LBRdckDZt2mTVqlX5+te/nueffz69e/fOaaedlrvuuitdunQpzr/hhhvStm3bjBs3Llu2bMmwYcMyb968tGnTZs8XCgAAALCXlRTAevLJJ193Tvv27XP++eeXcnsAAAAAoMzK0dMrFAqveX3Hjh3zwx/+8HWf06FDh8yePTuzZ89+3bkAAAAA+7pDSrnotttuy7e+9a0dxr/1rW/l9ttv3+OiAAAAAIDy0tMDAAAAaB0lBbCuv/769OjRY4fxnj17Ztq0aXtcFAAAAABQXnp6AAAAAK2jpADW008/nX79+u0w3rdv3zzzzDN7XBQAAAAAUF56egAAAACto6QAVs+ePfOrX/1qh/Ff/vKX6d69+x4XBQAAAACUl54eAAAAQOsoKYD1gQ98IJ/85Cfz4x//ONu3b8/27dvz0EMP5VOf+lQ+8IEPlLtGAAAAAGAP6ekBAAAAtI62pVx07bXX5umnn86wYcPStu1fb/Hyyy/nox/9aKZNm1bWAgEAAACAPaenBwAAANA6SgpgtW/fPnfddVc+//nP55e//GU6duyYgQMHpm/fvuWuDwAAAAAoAz09AAAAgNZRUgDrFQMGDMiAAQPKVQsAAAAA0Mr09AAAAADKq6QA1vbt2zNv3rz86Ec/SlNTU15++eUW5x966KGyFAcAAAAAlIeeHgAAAEDrKCmA9alPfSrz5s3Le97zntTX16eqqqrcdQEAAAAAZaSnBwAAANA6SgpgLViwIN/85jfz7ne/u9z1AAAAAACtQE8PAAAAoHUcUspF7du3z5FHHlnuWgAAAACAVqKnBwAAANA6SgpgTZo0KTfddFMKhUK56wEAAAAAWoGeHgAAAEDrKOknCJcsWZIf//jHue+++/L2t7897dq1a3H+O9/5TlmKAwAAAADKQ08PAAAAoHWUFMA67LDDcs4555S7FgAAAACglejpAQAAALSOkgJYt912W7nrAAAAAABakZ4eAAAAQOs4pNQLX3rppTz44IO55ZZbsnHjxiTJH//4x2zatKlsxQEAAAAA5aOnBwAAAFB+Je2A9fTTT+fMM8/MM888k+bm5owYMSJdunTJjBkz8uKLL+YrX/lKuesEAAAAAPaAnh4AAABA6yhpB6xPfepTGTx4cNatW5eOHTsWx88555z86Ec/KltxAAAAAEB56OkBAAAAtI6SdsBasmRJfvrTn6Z9+/Ytxvv27Zs//OEPZSkMAAAAACgfPT0AAACA1lHSDlgvv/xytm/fvsP4s88+my5duuxxUQAAAABAeenpAQAAALSOkgJYI0aMyI033lj8XFVVlU2bNuXf//3f8+53v7tctQEAAAAAZaKnBwAAANA6Sgpg3XDDDVm8eHGOPvrovPjiiznvvPPylre8JX/4wx/yhS98YbfvM3369Bx//PHp0qVLevbsmbPPPjtPPPFEizmFQiFTp05NXV1dOnbsmKFDh+axxx5rMae5uTnjx49Pjx490rlz55x11ll59tlnS1kaAAAAAByQytXTAwAAAKClkgJYdXV1WblyZSZPnpyLL744xx13XK6//vr84he/SM+ePXf7PosXL85ll12WRx99NIsWLcpLL72UkSNHZvPmzcU5M2bMyKxZszJnzpwsW7YstbW1GTFiRDZu3FicM2HChCxcuDALFizIkiVLsmnTpowePXqnW6oDAAAAwMGoXD09AAAAAFqqKhQKhUoX8Yo///nP6dmzZxYvXpxTTjklhUIhdXV1mTBhQq688sokf93tqlevXvnCF76Qiy++OOvXr8+b3vSm3HHHHTn33HOTJH/84x/Tp0+f3HvvvTnjjDNe97kbNmxITU1N1q9fn65du7bqGjkwfemShypdwkHtsq+cXukSAAAAdoseBJTHgfBdmnnu6EqXcFCbdNf3K10CAAAA+4Hd7UG0LeXmX//611/z/Ec/+tFSbpv169cnSbp165YkWbNmTRobGzNy5MjinOrq6px66qlZunRpLr744qxYsSLbtm1rMaeuri719fVZunTpTgNYzc3NaW5uLn7esGFDSfUCAAAAwP6itXp6AAAAAAe7kgJYn/rUp1p83rZtW1544YW0b98+nTp1KqlZUygUMnHixJx88smpr69PE/rtwgAAyENJREFUkjQ2NiZJevXq1WJur1698vTTTxfn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D8pPPw/K62Ds6dkBC/ay9evXJ0m6detW4Upg/3bZZZflPe95T4YPH17pUmC/d88992Tw4MF5//vfn549e+a4447LrbfeWumyYL928skn50c/+lF+85vfJEl++ctfZsmSJXn3u99d4cpg/7dmzZo0NjZm5MiRxbHq6uqceuqpWbp0aQUrA+BApZ8H5aOnB+Whnwflp58Hretg6Om1rXQBcDApFAqZOHFiTj755NTX11e6HNhvLViwIP/93/+dZcuWVboUOCD8/ve/z80335yJEyfmqquuys9//vN88pOfTHV1dT760Y9WujzYL1155ZVZv359jjrqqLRp0ybbt2/Pddddlw9+8IOVLg32e42NjUmSXr16tRjv1atXnn766UqUBMABTD8PykdPD8pHPw/KTz8PWtfB0NMTwIK96PLLL8+vfvWrLFmypNKlwH5r7dq1+dSnPpUHHnggHTp0qHQ5cEB4+eWXM3jw4EybNi1Jctxxx+Wxxx7LzTffrGEDJbrrrrsyf/783HnnnXn729+elStXZsKECamrq8v5559f6fLggFBVVdXic6FQ2GEMAPaUfh6Uh54elJd+HpSffh7sHQdyT08AC/aS8ePH55577skjjzySI444otLlwH5rxYoVaWpqyqBBg4pj27dvzyOPPJI5c+akubk5bdq0qWCFsP/p3bt3jj766BZj//AP/5Bvf/vbFaoI9n//8i//ks985jP5wAc+kCQZOHBgnn766UyfPl3DBvZQbW1tkr++Nde7d+/ieFNT0w5v0AHAntDPg/LR04Py0s+D8tPPg9Z1MPT0Dql0AXCgKxQKufzyy/Od73wnDz30UPr161fpkmC/NmzYsKxatSorV64sHoMHD86HPvShrFy5UqMGSvCud70rTzzxRIux3/zmN+nbt2+FKoL93wsvvJBDDmn5x602bdrk5ZdfrlBFcODo169famtrs2jRouLY1q1bs3jx4gwZMqSClQFwoNDPg/LT04Py0s+D8tPPg9Z1MPT07IAFreyyyy7LnXfeme9+97vp0qVL8bdNa2pq0rFjxwpXB/ufLl26pL6+vsVY586d07179x3Ggd3z6U9/OkOGDMm0adMybty4/PznP89Xv/rVfPWrX610abDfGjNmTK677rq8+c1vztvf/vb84he/yKxZs/Lxj3+80qXBfmHTpk357W9/W/y8Zs2a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jX3cFQcAAKDcKisrU11dnSSprq7ORRddlM985jP561//mq233nqV+StWrMjXv/71PPTQQ6mvr8+2226bwYMH59xzzy3OOfXUU/PPf/4z+++/f6677rosW7YsJ5xwQsaMGZPWrVsnSebNm5fTTz89Dz74YKqrq/Od73znQ2v95je/mddeey0vvfRSampqiuM9e/bMF7/4xWy22WbFsbfffjunnXZafvazn2XLLbfMt771rXz9619P8t5rFXv06JGf/vSnueGGG/LYY49l/PjxOeWUU/Kd73wnN910U/76179m5513ztVXX53Pfe5zq5w3duzYPP744+nVq1d+8pOfZMGCBfmv//qv/O///m/233//3HrrrcXvb/r06fnmN7+ZJ598MsuXL8+ee+6Z66+/PnvttVeS957gS5Jjjz02SbLddtvltddeS11dXe65557i6x8/yvcKUIp1/RQ3jfn+AQCApjJgwIC88sorGTlyZEaNGrXGeb/4xS9y2WWX5ZVXXkm3bt1y9tlnZ+jQocXj22+/fb761a/mlVdeKa7aftBBB2XIkCG57bbbMnTo0MyZMyeHHXZYJk6cmJ///Of59re/nQULFuRLX/pSxowZk5YtWyZJbrvttowZMyYvvvhi2rdvn89+9rMZM2ZMunbt+rHu7Ywzzsj48ePzm9/8Jocddthq5/Tr1y977rlnxowZUxw75phjssUWW2TChAlJkoaGhlx66aW54447Mm/evGy77ba5+OKLc/rpp6/2mtOmTcvFF1+c6dOnZ6uttsqxxx6bkSNHpn379kmS+fPn59xzz80vf/nLNDQ05MADD8z3v//91NbWrvZ6/97zaw5WwAIAAIAyWrx4cX7yk59kxx13XONTWCtXrsw222yTO++8M88//3wuu+yyfPOb38ydd97ZaN7vf//7vPrqq/n973+fiRMnZsKECcUmR/JemOi1117LQw89lJ///Oe54YYbMm/evDXWtnLlykyaNCknnXRSo/DV+zbffPO0avX/Pdt13XXXZe+9986TTz6ZwYMHF8NR/+qiiy7KOeeckxdeeCGHHHJIvve97+W6667Ld7/73Tz99NM55JBDctRRR+Xll19udN63v/3tfOtb38oTTzyRVq1a5Ytf/GIuvPDCfO9738ujjz6aV199NZdddllx/qJFi3LKKafk0UcfzWOPPZba2tocdthhWbRoUZL3AlpJcsstt2Tu3LnF/dX5sO8VAAAAgE1Xy5YtM2LEiIwdOzZvvPHGaufMmDEjxx13XE444YQ888wzqaury6WXXrpKj+naa69Nr169MmPGjFx66aVJ3nvo8fvf/34mTZqU+++/Pw8//HAGDRqU3/zmN/nNb36TW2+9NTfddFN+/vOfF6+zbNmyXHnllXnqqadyzz33ZNasWTn11FM/9r1tv/32+cY3vpHhw4dn5cqVH/v895188smZNGlSvv/97+eFF17IjTfemM0333y1c5955pkccsghGTRoUJ5++un89Kc/zdSpU3PWWWcV55x66ql5/PHHc++99+aPf/xjCoVCDjvssLKuuG8FLGCDZwWy8rICGQDAx/erX/2q2GBYsmRJunXrll/96ldp0WL1z0m1bt06l19+eXG/R48emTZtWu68884cd9xxxfEtt9wy48aNS8uWLbPTTjvl8MMPz+9+97t87Wtfy0svvZT77rsvjz32WPbdd98kyc0335ydd955jXX+7W9/y/z587PTTjt9pPs67LDDMnjw4CTvBa2uv/76PPzww43OHzJkSAYNGlTc/+53v5uLLrooJ5xwQpLkmmuuye9///uMGTMmP/jBD4rzhg0blkMOOSRJcu655+aLX/xifve732W//fZLkpx++umNGlaf/exnG9X2wx/+MFtuuWWmTJmSI444orhS1hZbbFFcjWxNPuh7BQAAAIBjjz02e+65Z7797W/n5ptvXuX46NGj079//2KoqmfPnnn++edz7bXXNgpGffazn82wYcOK+1OnTs3y5cszfvz4fOITn0iSfP7zn8+tt96a//u//8vmm2+eXXbZJQcddFB+//vf5/jjj0+SnHbaacVr7LDDDvn+97+fT33qU1m8ePEag09r8q1vfSu33HJLfvKTn+TLX/7yxzo3SV566aXceeedmTx5cgYMGFCsaU2uvfbanHjiiRkyZEiSpLa2Nt///vdz4IEHZvz48ZkzZ07uvffe/OEPf0jfvn2TJD/5yU/SvXv33HPPPfnCF77wsWtsClbAAgAAgHXsoIMOysyZMzNz5sz86U9/ysCBA3PooYdm9uzZazznxhtvzN57752tt946m2++eX70ox/l9ddfbzRn1113LS4zniTdunUrrnD1wgsvpFWrVtl7772Lx3faaadsscUWa/zMQqGQJKmoqPhI97X77rsX/7mioiLV1dWrrLD1r5+/cOHCvPXWW8UQ1fv222+/vPDCC2u8dlVVVZJkt912azT2r581b968fOMb30jPnj3TqVOndOrUKYsXL17lO/soPuh7BQAAAIDkvQcLJ06cmOeff36VYy+88MJqe2Avv/xyVqxYURz7197Z+9q1a1cMXyXv9cG23377RkGqf++NPfnkkzn66KOz3XbbpUOHDunXr1+SlNQb23rrrTNs2LBcdtllWbZs2cc+f+bMmWnZsmUOPPDAjzR/xowZmTBhQjbffPPidsghh2TlypWZNWtWsc/5/kOmSdKlS5d88pOfXKWnuC4JYAEAAMA61r59++y4447Zcccd86lPfSo333xzlixZkh/96EernX/nnXfmvPPOy2mnnZYHHnggM2fOzFe+8pVVGh6tW7dutF9RUVFcGvzjhqmS95orW2655UduXHzQ57+vffv2q5z37zUVCoVVxv712u8f+/exf/2sU089NTNmzMiYMWMybdq0zJw5M126dCmpSfRR7gsAAACATdtnPvOZHHLIIfnmN7+5yrHV9bve79f9q9X1zlbXm/qgftWSJUsycODAbL755rntttsyffr03H333UlSUm8sSc4///wsXbo0N9xwwyrHWrRoscq9/OurANu2bfuxPmvlypU544wzig+wzpw5M0899VRefvnlfOITn1jt95as/jtelwSwAAAAoMwqKirSokWLLF26dLXHH3300fTt2zeDBw9O7969s+OOO+bVV1/9WJ+x88475913383jjz9eHHvxxRfzz3/+c43ntGjRIscff3x+8pOf5K233lrl+JIlS/Luu+9+rDr+VceOHVNTU5OpU6c2Gp82bdoHvhrxo3j00Udzzjnn5LDDDsuuu+6aysrK/O1vf2s0p3Xr1o2eMAQAAACAtXH11Vfnl7/8ZaZNm9ZofJdddlltD6xnz56NVl5vCv/7v/+bv/3tb7n66qtzwAEHZKeddlrr1dw333zzXHrppbnqqquycOHCRse23nrrzJ07t7i/YsWKPPvss8X93XbbLStXrsyUKVM+0mfttddeee6554oPsP7r1qZNm+yyyy55991386c//al4zt///ve89NJLa91TXBsCWAAAALCONTQ0pL6+PvX19XnhhRdy9tlnZ/HixTnyyCNXO3/HHXfM448/nt/+9rd56aWXcumll2b69Okf6zM/+clP5nOf+1y+9rWv5U9/+lNmzJiRr371qx/6BNqIESPSvXv37Lvvvvnxj3+c559/Pi+//HL++7//O3vuuWcWL178ser4dxdccEGuueaa/PSnP82LL76Yiy++ODNnzsy55567Vtfdcccdc+utt+aFF17In/70p5x00kmr3Ov222+f3/3ud6mvr8/8+fPX6vMAAAAAYLfddstJJ52UsWPHNhofOnRofve73+XKK6/MSy+9lIkTJ2bcuHEZNmxYk9ew7bbbpk2bNhk7dmz+8pe/5N57782VV1651tf9+te/nk6dOuWOO+5oNP7Zz342v/71r/PrX/86//u//5vBgwc3euhz++23zymnnJLTTjst99xzT2bNmpWHH344d95552o/56KLLsof//jHnHnmmZk5c2Zefvnl3HvvvTn77LOTJLW1tTn66KPzta99LVOnTs1TTz2VL33pS/l//p//J0cfffRa32epyhrAeuSRR3LkkUempqYmFRUVueeee1aZ88ILL+Soo45Kp06d0qFDh/zHf/xHo3dSNjQ05Oyzz85WW22V9u3b56ijjsobb7yxDu8CAAAAPp77778/3bp1S7du3bLvvvtm+vTp+dnPfpZ+/fqtdv43vvGNDBo0KMcff3z23Xff/P3vf8/gwYM/9ufecsst6d69ew488MAMGjQoX//619O1a9cPPGfLLbfMY489li996Uv5zne+k969e+eAAw7IHXfckWuvvTadOnX62HX8q3POOSdDhw7N0KFDs9tuu+X+++/Pvffem9ra2rW67n//939n/vz56d27d7785S/nnHPOWeVer7vuukyePDndu3dP79691+rzAAAAACBJrrzyylVek7fXXnvlzjvvzKRJk9KrV69cdtllueKKK3Lqqac2+edvvfXWmTBhQn72s59ll112ydVXX53vfve7a33d1q1b58orr8w777zTaPy0007LKaeckpNPPjkHHnhgevTokYMOOqjRnPHjx+fzn/98Bg8enJ122ilf+9rXsmTJktV+zu67754pU6bk5ZdfzgEHHJDevXvn0ksvTbdu3YpzbrnllvTp0ydHHHFEPv3pT6dQKOQ3v/nNKq9mXJcqCmt6OeI6cN999+UPf/hD9tprr/znf/5n7r777hxzzDHF46+++mo+9alP5fTTT88Xv/jFdOrU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hw5Nly5d0rlz5wwbNiy77bZbBgwYsE7vBQAAAAAAAAAA2PSUNYD1+OOP56CDDirun3/++UmSU045JRMmTPhI17j++uvTqlWrHHfccVm6dGn69++fCRMmpGXLls1RMgAAAAAAAAAAQFFZA1j9+vVLoVD4yPNfe+21VcY222yzjB07NmPHjm3CygAAAAAAAAAAAD5ci3IXAAAAAAAAAAAAsKESwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKJIAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJyhrAeuSRR3LkkUempqYmFRUVueeee4rHli9fnosuuii77bZb2rdvn5qampx88sl56623Gl2joaEhZ599drbaaqu0b98+Rx11VN544411fCcAAAAAAAAAAMCmqKwBrCVLlmSPPfbIuHHjVjn29ttv54knnsill16aJ554InfddVdeeumlHHXUUY3mDRkyJHfffXcmTZqUqVOnZvHixTniiCOyYsWKdXUbAAAAAAAAAADAJqpVOT/80EMPzaGHHrraY506dcrkyZMbjY0dOzaf+tSn8vrrr2fbbbfNggULcvPNN+fWW2/NgAEDkiS33XZbunfvngcffDCHHHJIs98DAAAAAAAAAACw6SrrClgf14IFC1JRUZEtttgiSTJjxowsX748AwcOLM6pqalJr169Mm3atDVep6GhIQsXLmy0AQAAAAAAAAAAfFwbTADrnXfeycUXX5wTTzwxHTt2TJLU19enTZs22XLLLRvNraqqSn19/RqvNXLkyHTq1Km4de/evVlrBwAAAAAAAAAANk4bRABr+fLlOeGEE7Jy5crccMMNHzq/UCikoqJijceHDx+eBQsWFLc5c+Y0ZbkAAAAAAAAAAMAmYr0PYC1fvjzHHXdcZs2alcmTJxdXv0qS6urqLFu2LPPnz290zrx581JVVbXGa1ZWVqZjx46NNgAAAAAAAAAAgI9rvQ5gvR++evnll/Pggw+mS5cujY736dMnrVu3zuTJk4tjc+fOzbPPPpu+ffuu63IBAAAAAAAAAIBNTKtyfvjixYvzyiuvFPdnzZqVmTNnpnPnzqmpqcnnP//5PPHEE/nVr36VFStWpL6+PknSuXPntGnTJp06dcrpp5+eoUOHpkuXLuncuXOGDRuW3XbbLQMGDCjXbQEAAAAAAAAAAJuIsgawHn/88Rx00EHF/fPPPz9Jcsopp6Suri733ntvkmTPPfdsdN7vf//79OvXL0ly/fXXp1WrVjnuuOOydOnS9O/fPxMmTEjLli3XyT0AAAAAAAAAAACbrrIGsPr165dCobDG4x907H2bbbZZxo4dm7FjxzZlaQAAAAAAAAAAAB+qRbkLAAAAAAAAAAAA2FAJYAEAAAAAAAAAAJRIAAsAAAAAAAAAAKBEAlgAAAAAwEcyfvz47L777unYsWM6duyYT3/607nvvvuKxwuFQurq6lJTU5O2bdumX79+ee655xpdo6GhIWeffXa22mqrtG/fPkcddVTeeOONdX0rAAAAAE1GAAsAAAAA+Ei22WabXH311Xn88cfz+OOP57Of/WyOPvroYshq1KhRGT16dMaNG5fp06enuro6Bx98cBYtWlS8xpAhQ3L33Xdn0qRJmTp1ahYvXpwjjjgiK1asKNdtAQAAAKwVASwAAAAA4CM58sgjc9hhh6Vnz57p2bNnrrrqqmy++eZ57LHHUigUMmbMmFxyySUZNGhQevXqlYkTJ+btt9/O7bffniRZsGBBbr755lx33XUZMGBAevfundtuuy3PPPNMHnzwwTLfHQAAAEBpBLAAAAAAgI9txYoVmTRpUpYsWZJPf/rTmTVrVurr6zNw4MDinMrKyhx44IGZNm1akmTGjBlZvnx5ozk1NTXp1atXcc7qNDQ0ZOHChY02AAAAgPWFABYAAAAA8JE988wz2XzzzVNZWZlvfOMbufvuu7PLLrukvr4+SVJVVdVoflVVVfFYfX192rRpky233HKNc1Zn5MiR6dSpU3Hr3r17E98VAAAAQOkEsAAAAACAj+yTn/xkZs6cmcceeyz/9V//lVNOOSXPP/988XhFRUWj+YVCYZWxf/dhc4YPH54FCxYUtzlz5qzdTQAAAAA0IQEsAAAAAOAja9OmTXbcccfsvffeGTlyZPbYY49873vfS3V1dZKsspLVvHnziqtiVVdXZ9myZZk/f/4a56xOZWVlOnbs2GgDAAAAWF8IYAEAAAAAJSsUCmloaEiPHj1SXV2dyZMnF48tW7YsU6ZMSd++fZMkffr0SevWrRvNmTt3bp599tniHAAAAIANTatyFwAAAAAAbBi++c1v5tBDD0337t2zaNGiTJo0KQ8//HDuv//+VFRUZMiQIRkxYkRqa2tTW1ubESNGpF27djnxxBOTJJ06dcrpp5+eoUOHpkuXLuncuXOGDRuW3XbbLQMGDCjz3QEAAACURgALAAAAAPhI/u///i9f/vKXM3fu3HTq1Cm777577r///hx88MFJkgsvvDBLly7N4MGDM3/+/Oy777554IEH0qFDh+I1rr/++rRq1SrHHXdcli5dmv79+2fChAlp2bJluW4LAAAAYK0IYAEAAAAAH8nNN9/8gccrKipSV1eXurq6Nc7ZbLPNMnbs2IwdO7aJqwMAAAAojxblLgAAAAAAAAAAAGBDJYAFAAAAAAAAAABQIgEsAAAAAAAAAACAEglgAQAAAAAAAAAAlEgACwAAAAAAAAAAoEQCWAAAAAAAAAAAACUSwAIAAAAAAAAAACiRABYAAAAAAAAAAECJBLAAAAAAAAAAAABKVFIAa9asWU1dBwAAAADQjPT0AAAAAJpHSQGsHXfcMQcddFBuu+22vPPOO01dEwAAAADQxPT0AAAAAJpHSQGsp556Kr17987QoUNTXV2dM844I3/+85+bujYAAAAAoIno6QEAAAA0j5ICWL169cro0aPz5ptv5pZbbkl9fX3233//7Lrrrhk9enT++te/NnWdAAAAAMBa0NMDAAAAaB4lBbDe16pVqxx77LG58847c8011+TVV1/NsGHDss022+Tkk0/O3LlzP/D8Rx55JEceeWRqampSUVGRe+65p9HxQqGQurq61NTUpG3btunXr1+ee+65RnMaGhpy9tlnZ6uttkr79u1z1FFH5Y033lib2wIAAACAjdba9vQAAAAAaGytAliPP/54Bg8enG7dumX06NEZNmxYXn311Tz00EN58803c/TRR3/g+UuWLMkee+yRcePGrfb4qFGjMnr06IwbNy7Tp09PdXV1Dj744CxatKg4Z8iQIbn77rszadKkTJ06NYsXL84RRxyRFStWrM2tAQAAAMBGaW17egAAAAA01qqUk0aPHp1bbrklL774Yg477LD8+Mc/zmGHHZYWLd7Lc/Xo0SM//OEPs9NOO33gdQ499NAceuihqz1WKBQyZsyYXHLJJRk0aFCSZOLEiamqqsrtt9+eM844IwsWLMjNN9+cW2+9NQMGDEiS3HbbbenevXsefPDBHHLIIaXcHgAAAABsdJqqpwcAAABAYyWtgDV+/PiceOKJef3113PPPffkiCOOKDZq3rftttvm5ptvLrmwWbNmpb6+PgMHDiyOVVZW5sADD8y0adOSJDNmzMjy5csbzampqUmvXr2Kc1anoaEhCxcubLQBAAAAwMZsXfT0AAAAADZFJa2A9fLLL3/onDZt2uSUU04p5fJJkvr6+iRJVVVVo/GqqqrMnj27OKdNmzbZcsstV5nz/vmrM3LkyFx++eUl1wYAAAAAG5p10dMDAAAA2BSVtALWLbfckp/97GerjP/sZz/LxIkT17qof1VRUdFov1AorDL27z5szvDhw7NgwYLiNmfOnCapFQAAAADWV+uypwcAAACwKSkpgHX11Vdnq622WmW8a9euGTFixFoXlSTV1dVJsspKVvPmzSuuilVdXZ1ly5Zl/vz5a5yzOpWVlenYsWOjDQAAAAA2ZuuipwcAAACwKSopgDV79uz06NFjlfHtttsur7/++loXlSQ9evRIdXV1Jk+eXBxbtmxZpkyZkr59+yZJ+vTpk9atWzeaM3fu3Dz77LPFOQAAAADAuunpAQAAAGyKWpVyUteuXfP0009n++23bzT+1FNPpUuXLh/5OosXL84rr7xS3J81a1ZmzpyZzp07Z9ttt82QIUMyYsSI1NbWpra2NiNGjEi7du1y4oknJkk6deqU008/PUOHDk2XLl3SuXPnDBs2LLvttlsGDBhQyq0BAAAAwEapqXp6AAAAADRWUgDrhBNOyDnnnJMOHTrkM5/5TJJkypQpOffcc3PCCSd85Os8/vjjOeigg4r7559/fpLklFNOyYQJE3LhhRdm6dKlGTx4cObPn5999903DzzwQDp06FA85/rrr0+rVq1y3HHHZenSpenfv38mTJiQli1blnJrAAAAALBRaqqeHgAAAACNVRQKhcLHPWnZsmX58pe/nJ/97Gdp1eq9DNfKlStz8skn58Ybb0ybNm2avNDmtHDhwnTq1CkLFixIx44dy11OSd64+NFylwBsora5+oBylwAAABuMjaEHwYZrY+rpbQy/pbq6unKXsEnz/QMAAPBRfNQeREkrYLVp0yY//elPc+WVV+app55K27Zts9tuu2W77bYruWAAAAAAoPno6QEAAAA0j5ICWO/r2bNnevbs2VS1AAAAAADNTE8PAAAAoGmVFMBasWJFJkyYkN/97neZN29eVq5c2ej4Qw891CTFAQAAAABNQ08PAAAAoHmUFMA699xzM2HChBx++OHp1atXKioqmrouAAAAAKAJ6ekBAAAANI+SAliTJk3KnXfemcMOO6yp6wEAAAAAmoGeHgAAAEDzaFHKSW3atMmOO+7Y1LUAAAAAAM1ETw8AAACgeZQUwBo6dGi+973vpVAoNHU9AAAAAEAz0NMDAAAAaB4lvYJw6tSp+f3vf5/77rsvu+66a1q3bt3o+F133dUkxQEAAAAATUNPDwAAAKB5lBTA2mKLLXLsscc2dS0AAAAAQDPR0wMAAABoHiUFsG655ZamrgMAAAAAaEZ6egAAAADNo0WpJ7777rt58MEH88Mf/jCLFi1Kkrz11ltZvHhxkxUHAAAAADQdPT0AAACAplfSClizZ8/O5z73ubz++utpaGjIwQcfnA4dOmTUqFF55513cuONNzZ1nQAAAADAWtDTAwAAAGgeJa2Ade6552bvvffO/Pnz07Zt2+L4sccem9/97ndNVhwAAAAA0DT09AAAAACaR0krYE2dOjV/+MMf0qZNm0bj2223Xd58880mKQwAAAAAaDp6egAAAADNo6QVsFauXJkVK1asMv7GG2+kQ4cOa10UAAAAANC09PQAAAAAmkdJAayDDz44Y8aMKe5XVFRk8eLF+fa3v53DDjusqWoDAAAAAJqInh4AAABA8yjpFYTXX399DjrooOyyyy555513cuKJJ+bll1/OVlttlTvuuKOpawQAAAAA1pKeHgAAAEDzKCmAVVNTk5kzZ+aOO+7IE088kZUrV+b000/PSSedlLZt2zZ1jQAAAADAWtLTAwAAAGgeJQWwkqRt27Y57bTTctpppzVlPQAAAABAM9HTAwAAAGh6JQWwfvzjH3/g8ZNPPrmkYgAAAACA5qGnBwAAANA8SgpgnXvuuY32ly9fnrfffjtt2rRJu3btNGsAAAAAYD2jpwcAAADQPFqUctL8+fMbbYsXL86LL76Y/fffP3fccUdT1wgAAAAArCU9PQAAAIDmUVIAa3Vqa2tz9dVXr/IkHQAAAACwftLTAwAAAFh7TRbASpKWLVvmrbfeaspLAgAAAADNSE8PAAAAYO20KuWke++9t9F+oVDI3LlzM27cuOy3335NUhgAAAAA0HT09AAAAACaR0kBrGOOOabRfkVFRbbeeut89rOfzXXXXdcUdQEAAAAATUhPDwAAAKB5lBTAWrlyZVPXAQAAAAA0Iz09AAAAgObRotwFfJB333033/rWt9KjR4+0bds2O+ywQ6644opGzaJCoZC6urrU1NSkbdu26devX5577rkyVg0AAAAAAAAAAGwqSloB6/zzz//Ic0ePHl3KRyRJrrnmmtx4442ZOHFidt111zz++OP5yle+kk6dOuXcc89NkowaNSqjR4/OhAkT0rNnz3znO9/JwQcfnBdffDEdOnQo+bMBAAAAYGOyrnp6AAAAAJuakgJYTz75ZJ544om8++67+eQnP5kkeemll9KyZcvstddexXkVFRVrVdwf//jHHH300Tn88MOTJNtvv33uuOOOPP7440neW/1qzJgxueSSSzJo0KAkycSJE1NVVZXbb789Z5xxxlp9PgAAAABsLNZVTw8AAABgU1NSAOvII49Mhw4dMnHixGy55ZZJkvnz5+crX/lKDjjggAwdOrRJitt///1z44035qWXXkrPnj3z1FNPZerUqRkzZkySZNasWamvr8/AgQOL51RWVubAAw/MtGnT1hjAamhoSENDQ3F/4cKFTVIvAAAAAKyv1lVPDwAAAGBTU1IA67rrrssDDzxQbNQkyZZbbpnvfOc7GThwYJM1ay666KIsWLAgO+20U1q2bJkVK1bkqquuyhe/+MUkSX19fZKkqqqq0XlVVVWZPXv2Gq87cuTIXH755U1SIwAAAABsCNZVTw8AAABgU9OilJMWLlyY//u//1tlfN68eVm0aNFaF/W+n/70p7ntttty++2354knnsjEiRPz3e9+NxMnTmw079+XRS8UCh+4VPrw4cOzYMGC4jZnzpwmqxkAAAAA1kfrqqcHAAAAsKkpaQWsY489Nl/5yldy3XXX5T/+4z+SJI899lguuOCCDBo0qMmKu+CCC3LxxRfnhBNOSJLstttumT17dkaOHJlTTjkl1dXVSd5bCatbt27F8+bNm7fKqlj/qrKyMpWVlU1WJwAAAACs79ZVTw8AAABgU1NSAOvGG2/MsGHD8qUvfSnLly9/70KtWuX000/Ptdde22TFvf3222nRovEiXS1btszKlSuTJD169Eh1dXUmT56c3r17J0mWLVuWKVOm5JprrmmyOgAAAABgQ7euenoAAAAAm5qSAljt2rXLDTfckGuvvTavvvpqCoVCdtxxx7Rv375JizvyyCNz1VVXZdttt82uu+6aJ598MqNHj85pp52W5L1XDw4ZMiQjRoxIbW1tamtrM2LEiLRr1y4nnnhik9YCAAAAABuyddXTAwAAANjUtPjwKWs2d+7czJ07Nz179kz79u1TKBSaqq4kydixY/P5z38+gwcPzs4775xhw4bljDPOyJVXXlmcc+GFF2bIkCEZPHhw9t5777z55pt54IEH0qFDhyatBQAAAAA2BmvT0xs5cmT22WefdOjQIV27ds0xxxyTF198sdGcQqGQurq61NTUpG3btunXr1+ee+65RnMaGhpy9tlnZ6uttkr79u1z1FFH5Y033miS+wMAAABY10oKYP39739P//7907Nnzxx22GGZO3dukuSrX/1qhg4d2mTFdejQIWPGjMns2bOzdOnSvPrqq/nOd76TNm3aFOdUVFSkrq4uc+fOzTvvvJMpU6akV69eTVYDAAAAAGwMmqKnN2XKlJx55pl57LHHMnny5Lz77rsZOHBglixZUpwzatSojB49OuPGjcv06dNTXV2dgw8+OIsWLSrOGTJkSO6+++5MmjQpU6dOzeLFi3PEEUdkxYoVTXvTAAAAAOtASQGs8847L61bt87rr7+edu3aFcePP/743H///U1WHAAAAADQNJqip3f//ffn1FNPza677po99tgjt9xyS15//fXMmDEjyXurX40ZMyaXXHJJBg0alF69emXixIl5++23c/vttydJFixYkJtvvjnXXXddBgwYkN69e+e2227LM888kwcffLDpbxwAAACgmZUUwHrggQdyzTXXZJtttmk0Xltbm9mzZzdJYQAAAABA02mOnt6CBQuSJJ07d06SzJo1K/X19Rk4cGBxTmVlZQ488MBMmzYtSTJjxowsX7680Zyampr06tWrOOffNTQ0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mtPQ0JCzzz47W221Vdq3b5+jjjoqb7zxxjq8CwAAAICmtV4HsGbMmJF58+alT58+adWqVVq1apUpU6bk+9//flq1alVc+er9lbDeN2/evFVWxfpXlZWV6dixY6MNAAAAAPhgS5YsyR577JFx48at9vioUaMyevTojBs3LtOnT091dXUOPvjgLFq0qDhnyJAhufvuuzNp0qRMnTo1ixcvzhFHHJEVK1asq9sAAAAAaFLr9SsI+/fvn2eeeabR2Fe+8pXstNNOueiii7LDDjukuro6kydPTu/evZMky5Yty5QpU3LNNdeUo2QAAAAA2GgdeuihOfTQQ1d7rFAoZMyYMbnkkksyaNCgJMnEiRNTVVWV22+/PWeccUYWLFiQm2++ObfeemsGDBiQJLntttvSvXv3PPjggznkkEPW2b0AAAAANJX1OoDVoUOH9OrVq9FY+/bt06VLl+L4kCFDMmLEiNTW1qa2tjYjRoxIu3btcuKJJ5ajZAAAAADYJM2aNSv19fUZOHBgcayysjIHHnhgpk2bljPOOCMzZszI8uXLG82pqalJr169Mm3atDUGsBoaGtLQ0FDcX7hwYfPdCAAAAMDHtF4HsD6KCy+8MEuXLs3gwYMzf/787LvvvnnggQfSoUOHcpcGAAAAAJuM+vr6JElVVVWj8aqqqsyePbs4p02bNtlyyy1XmfP++aszcuTIXH755U1cMQAAAEDT2OACWA8//HCj/YqKitTV1aWurq4s9QAAAAAA/5+KiopG+4VCYZWxf/dhc4YPH57zzz+/uL9w4cJ079597QoFAAAAaCItyl0AAAAAALDhq66uTpJVVrKaN29ecVWs6urqLFu2LPPnz1/jnNWprKxMx44dG20AAAAA6wsBLAAAAABgrfXo0SPV1dWZPHlycWzZsmWZMmVK+vbtmyTp06dPWrdu3WjO3Llz8+yzzxbnAAAAAGxoNrhXEAIAAAAA5bF48eK88sorxf1Zs2Zl5syZ6dy5c7bddtsMGTIkI0aMSG1tbWprazNixIi0a9cuJ554YpKkU6dOOf300zN06NB06dIlnTt3zrBhw7LbbrtlwIAB5botAAAAgLUigAUAAAAAfCSPP/54DjrooOL++eefnyQ55ZRTMmHChFx44YVZunRpBg8enPnz52fffffNAw88kA4dOhTPuf7669OqVascd9xxWbp0afr3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9fM2+ADW+yoqKhrtFwqFVcbeN3z48Jx//vnF/ZUrV+Yf//hHunTpssZzoCksXLgw3bt3z5w5c9KxY8dylwMbPL8paHp+V9C0/Kag6W3ov6tCoZBFixalpqam3KVA2ennsSHY0P+9A+sbvyloen5X0LT8pqDpbei/q4/az9vgA1hbbbVVWrZsucrTcfPmzVvlKbr3VVZWprKystHYFlts0Vwlwio6duy4Qf4PC6yv/Kag6fldQdPym4KmtyH/rqx8xaZOP48N0Yb87x1YH/lNQdPzu4Km5TcFTW9D/l19lH5ei3VQR7Nq06ZN+vTpk8mTJzcanzx5cvr27VumqgAAAACA1dHPAwAAADY2G/wKWEly/vnn58tf/nL23nvvfPrTn85NN92U119/Pd/4xjfKXRoAAAAA8G/08wAAAICNyUYRwDr++OPz97//PVdccUXmzp2bXr165Te/+U222267cpcGjVRWVubb3/72KkvmA6Xxm4Km53cFTctvCpqe3xVsHPTz2FD49w40Lb8paHp+V9C0/Kag6W0qv6uKQqFQKHcRAAAAAAAAAAAAG6IW5S4AAAAAAAAAAABgQyWABQAAAAAAAAAAUCIBLAAAAAAAAAAAgBIJYAEAAAAAAAAAAJRIAAua2ciRI7PPPvukQ4cO6dq1a4455pi8+OKL5S4LNhojR45MRUVFhgwZUu5SYIP25ptv5ktf+lK6dOmSdu3aZc8998yMGTPKXRZssN59991861vfSo8ePdK2bdvssMMOueKKK7Jy5cpylwYbhEceeSRHHnlkampqUlFRkXvuuafR8UKhkLq6utTU1KRt27bp169fnnvuufIUC8BGRz8Pmp+eHqw9/TxoWvp5sPY29Z6eABY0sylTpuTMM8/MY489lsmTJ+fdd9/NwIEDs2TJknKXBhu86dOn56abbsruu+9e7lJggzZ//vzst99+ad26de677748//zzue6667LFFluUuzTYYF1zzTW58cYbM27cuLzwwgsZNWpUrr322owdO7bcpcEGYcmSJdljjz0ybty41R4fNWpURo8enXHjxmX69Omprq7OwQcfnEWLFq3jSgHYGOnnQfPS04O1p58HTU8/D9bept7TqygUCoVyFwGbkr/+9a/p2rVrpkyZks985jPlLgc2WIsXL85ee+2VG264Id/5zney5557ZsyYMeUuCzZIF198cf7whz/k0UcfLXcpsNE44ogjUlVVlZtvvrk49p//+Z9p165dbr311jJWBhueioqK3H333TnmmGOSvPekXE1NTYYMGZKLLrooSdLQ0JCqqqpcc801OeOMM8pYLQAbI/08aDp6etA09POg6ennQdPaFHt6VsCCdWzBggVJks6dO5e5EtiwnXnmmTn88MMzYMCAcpcCG7x77703e++9d77whS+ka9eu6d27d370ox+VuyzYoO2///753e9+l5deeilJ8tRTT2Xq1Kk57LDDylwZbPhmzZqV+vr6DBw4sDhWWVmZAw88MNOmTStjZQBsrPTzoOno6UHT0M+DpqefB81rU+jptSp3AbApKRQKOf/887P//vunV69e5S4HNliTJk3KE088kenTp5e7FNgo/OUvf8n48eNz/vnn55vf/Gb+/Oc/55xzzkllZWVOPvnkcpcHG6SLLrooCxYsyE477ZSWLVtmxYoVueqqq/LFL36x3KXBBq++vj5JUlVV1Wi8qqoqs2fPLkdJAGzE9POg6ejpQdPRz4Omp58HzWtT6OkJYME6dNZZZ+Xpp5/O1KlTy10KbLDmzJmTc889Nw888EA222yzcpcDG4WVK1dm7733zogRI5IkvXv3znPPPZfx48dr2ECJfvrTn+a2227L7bffnl133TUzZ87MkCFDUlNTk1NOOaXc5cFGoaKiotF+oVBYZQwA1pZ+HjQNPT1oWvp50PT082Dd2Jh7egJYsI6cffbZuffee/PII49km222KXc5sMGaMWNG5s2blz59+hTHVqxYkUceeSTjxo1LQ0NDWrZsWcYKYcPTrVu37LLLLo3Gdt555/ziF78oU0Ww4bvgggty8cUX54QTTkiS7Lbbbpk9e3ZGjhypYQNrqbq6Osl7T81169atOD5v3rxVnqADgLWhnwdNR08PmpZ+HjQ9/TxoXptCT69FuQuAjV2hUMhZZ52Vu+66Kw899FB69OhR7pJgg9a/f/8888wzmTlzZnHbe++9c9JJJ2XmzJkaNVCC/fbbLy+++GKjsZdeeinbbbddmSqCDd/bb7+dFi0a/+dWy5Yts3LlyjJVBBuPHj16pLq6OpMnTy6OLVu2LFOmTEnfvn3LWBkAGwv9PGh6enrQtPTzoOnp50Hz2hR6elbAgmZ25pln5vbbb8///M//pEOHDsV3m3bq1Clt27Ytc3Ww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9tJLL2W77bZLkvTo0SPV1dWZPHlyevfuneS9vyxTpkzJNddckyTp06dPWrduncmTJ+e4445LksydOzfPPvtsRo0atQ7vBgAAANat98NXXbt2Tbt27QR+yqxQKOTtt9/OvHnzkqTRw2QAAAAA0JwKhULmzp2bli1bpnv37mnRokW5S1rvNWU/r6wBrPPOOy99+/bNiBEjctxxx+XPf/5zbrrpptx0001J3nv14JAhQzJixIjU1tamtrY2I0aMSLt27XLiiScmSTp16pTTTz89Q4cOTZcuXdK5c+cMGzYsu+22WwYMGFDO2wMAAIBms2LFimL4qkuXLuUuh/+/tm3bJnlvZe6uXbt6HSEAAAAA68S7776bt99+OzU1NWnXrl25y9lgNFU/r6wBrH322Sd33313hg8fniuuuCI9evTImDFjctJJJxXnXHjhhVm6dGkGDx6c+fPnZ999980DDzyQDh06FOdcf/31adWqVY477rgsXbo0/fv3z4QJEzQ5AQAA2GgtX748STRT1kPv/5ksX75cbwIAAACAdWLFihVJkjZt2pS5kg1PU/TzKgqFQqEpi9oQLVy4MJ06dcqCBQvSsWPHcpcDAAAAH+qdd97JrFmz0qNHj2y22WblLod/8UF/NnoQ0DT8lgAAAKAx/cLSNUU/zwsfAQAAAAAAAAAASiSABQAAAKy3+vXrlyFDhpS7DAAAAACgjCoqKnLPPfeUu4w1alXuAgAAAICm9buHPrHOPqv/Z1/92OeceuqpmThxYs4444zceOONjY4NHjw448ePzymnnJIJEybkrrvuSuvWrYvHt99++wwZMkQoCwAAAAA+gurfz1ynn1d/0J6lnVdfn6uuuiq//vWv8+abb6Zr167Zc889M2TIkPTv379pi2wGVsACAAAA1rnu3btn0qRJWbp0aXHsnXfeyR133JFtt922ONa5c+d06NChHCUCAAAAAOvAa6+9lj59+uShhx7KqFGj8swzz+T+++/PQQcdlDPPPLPc5X0kVsDaSKzLp5tZVSlPfAMAAGzK9tprr/zlL3/JXXfdlZNOOilJctddd6V79+7ZYYcdivP69euXPffcM2PGjEm/fv0ye/bsnHfeeTnvvPOSJIVCIUnyi1/8IpdddlleeeWVdOvWLWeffXaGDh1avM4NN9yQ66+/PnPmzEmnTp1ywAEH5Oc//3nxGtdee21uvPHGzJ07Nz179syll16az3/+80mS+fPn56yzzsoDDzyQxYsXZ5tttsk3v/nNfOUrX1kn3xUATW9dPwFOY6U+EQ8AAGycBg8enIqKivz5z39O+/bti+O77rprTjvttNWec9FFF+Xuu+/OG2+8kerq6px00km57LLLiqvpP/XUUxkyZEgef/zxVFRUpLa2Nj/84Q+z9957N8s9CGABAAAAZfGVr3wlt9xySzGA9d///d857bTT8vDDD692/l133ZU99tgjX//61/O1r32tOD5jxowcd9xxqaury/HHH59p06Zl8ODB6dKlS0499dQ8/vjjOeecc3Lrrbemb9+++cc//pFHH320eP63vvWt3HXXXRk/fnxqa2vzyCOP5Etf+lK23nrrHHjggbn00kvz/PPP57777stWW22VV155pdHKXQAAAABAaf7xj3/k/vvvz1VXXdUofPW+LbbYYrXndejQIRMmTEhNTU2eeeaZfO1rX0uHDh1y4YUXJklOOumk9O7dO+PHj0/Lli0zc+bMYjirOQhgAQAAAGXx5S9/OcOHD89rr72WioqK/OEPf8ikSZPWGMDq3LlzWrZsmQ4dOqS6uro4Pnr06PTv3z+XXnppkqRnz555/vnnc+211+bUU0/N66+/nvbt2+eII45Ihw4dst1226V3795JkiVLlmT06NF56KGH8ulPfzpJssMOO2Tq1Kn54Q9/mAMPPDCvv/56evfuXXw6bvvtt2++LwUAAAAANiGvvPJKCoVCdtppp4913re+9a3iP2+//fYZOnRofvrTnxYDWK+//nouuOCC4nVra2ubrujVEMACAAAAymKrrbbK4YcfnokTJ6ZQKOTwww/PVltt9bGv88ILL+Too49uNLbffvtlzJgxWbFiRQ4++OBst9122WGHHfK5z30un/vc53LsscemXbt2ef755/POO+/k4IMPbnT+smXLiiGt//qv/8p//ud/5oknnsjAgQNzzDHHpG/fvqXfOAAAAACQJCkUCkmSioqKj3Xez3/+84wZMyavvPJKFi9enHfffTcdO3YsHj///PPz1a9+NbfeemsGDBiQL3zhC/nEJz7RpLX/qxbNdmUAAACAD3HaaadlwoQJmThxYk477bSSrlEoFFZp0LzfuEneW478iSeeyB133JFu3brlsssuyx577JF//vOfWblyZZLk17/+dWbOnFncnn/++fz85z9Pkhx66KGZPXt2hgwZkrfeeiv9+/fPsGHDSrxjAAAAAOB9tbW1qaioyAsvvPCRz3nsscdywgkn5NBDD82vfvWrPPnkk7nkkkuybNmy4py6uro899xzOfzww/PQQw9ll112yd13390ct5BEAAsAAAAoo8997nNZtmxZli1blkMOOeRD57dp0yYrVqxoNLbLLrtk6tSpjcamTZuWnj17pmXLlkmSVq1aZcCAARk1alSefvrpvPbaa8XGS2VlZV5//fXsuOOOjbbu3bsXr7f11lvn1FNPzW233ZYxY8bkpptuaoK7BwAAAIBNW+fOnXPIIYfkBz/4QZYsWbLK8X/+85+rjP3hD3/Idtttl0suuSR77713amtrM3v27FXm9ezZM+edd14eeOCBDBo0KLfccktz3EISryAEAAAAyqhly5bFp9veD0t9kO233z6PPPJITjjhhFRWVmarrbbK0KFDs88+++TKK6/M8ccfnz/+8Y8ZN25cbrjhhiTJr371q/zlL3/JZz7zmWy55Zb5zW9+k5UrV+aTn/xkOnTokGHDhuW8887LypUrs//++2fhwoWZNm1aNt9885xyyim57LLL0qdPn+y6665paGjIr371q+y8887N+r0AAAAAwKbihhtuSN++ffOpT30qV1xxRXbfffe8++67mTx5csaPH7/K6lg77rhjXn/99UyaNCn77LNPfv3rXzda3Wrp0qW54IIL8vnPfz49evTIG2+8kenTp+c///M/m+0eBLAAAACAsurYseNHnnvFFVfkjDPOyCc+8Yk0NDSkUChkr732yp133pnLLrssV155Zbp165Yrrrgip556apJkiy22yF133ZW6urq88847qa2tzR133JFdd901SXLllVema9euGTlyZP7yl79kiy22yF577ZVvfvObSd5bdWv48OF57bXX0rZt2xxwwAGZNGlSk38PAAAAALAp6tGjR5544olcddVVGTp0aObOnZutt946ffr0yfjx41eZf/TRR+e8887LWWedlYaGhhx++OG59NJLU1dXl+S9Bz3//ve/5+STT87//d//ZauttsqgQYNy+eWXN9s9VBQKhUKzXX0DsXDhwnTq1CkLFiz4WE3f9cnvHvpEuUvYpPX/7KvlLgEAANjEvPPOO5k1a1Z69OiRzTbbrNzl8C8+6M9mY+hBwPrAb4m1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x4J+67XXXst/+S//JaNHj86gQYPy7ne/O3/7t3+b119/vdLRoF948sknc9VVV6WpqSlVVVX53ve+1+t8oVDI/Pnz09TUlEGDBmXixInZtm1bZcICcNwxz4O+Z6YHR848D8rLPA+O3Ik+01PAgj62fv36zJw5M0899VTWrVuX1157LVOnTs3evXsrHQ36vU2bNuX+++/PueeeW+ko0K91dnbmAx/4QAYMGJCHHnooP/3pT3PXXXflne98Z6WjQb91xx135Otf/3qWLFmSf/7nf87ChQtz55135p577ql0NOgX9u7dm/POOy9Llix50/MLFy7MokWLsmTJkmzatCmNjY2ZMmVKdu/efZSTAnA8Ms+DvmWmB0fOPA/KzzwPjtyJPtOrKhQKhUqHgBPJv/zLv+SUU07J+vXr88EPfrDScaDf2rNnT973vvfla1/7Wm6//facf/75Wbx4caVjQb/0+c9/Pj/84Q/zv//3/650FDhuXHnllWloaMjSpUuLa//xP/7HDB48OA888EAFk0H/U1VVlTVr1uSaa65J8ruflGtqakpra2s+97nPJUl6enrS0NCQO+64IzfeeGMF0wJwPDLPg/Ix04PyMM+D8jPPg/I6EWd6noAFR1lXV1eSZPjw4RVOAv3bzJkz86EPfSiTJ0+udBTo9x588MGMGzcu/+k//aeccsopueCCC/KNb3yj0rGgX7vooovy6KOP5mc/+1mS5Cc/+Uk2bNiQK664osLJoP/bvn17Ojo6MnXq1OJabW1tLr744mzcuLGCyQA4XpnnQfmY6UF5mOdB+ZnnQd86EWZ6NZUOACeSQqGQ2bNn56KLLkpLS0ul40C/tXr16jzzzDPZtGlTpaPAceEXv/hF7r333syePTtf+MIX8qMf/Sh/+Zd/mdra2lx//fWVjgf90uc+97l0dXXlj//4j1NdXZ2DBw/mS1/6Uj7+8Y9XOhr0ex0dHUmShoaGXusNDQ156aWXKhEJgOOYeR6Uj5kelI95HpSfeR70rRNhpqeABUfRzTffnGeffTYbNmyodBTot3bs2JFbbrkla9euzcknn1zpOHBceP311zNu3Li0tbUlSS644IJs27Yt9957r4ENlOjb3/52Vq5cmVWrVuWcc87Jli1b0tramqamptxwww2VjgfHhaqqql6vC4XCIWsAcKTM86A8zPSgvMzzoPzM8+DoOJ5negpYcJTMmjUrDz74YJ588smcdtpplY4D/dbmzZuza9eujB07trh28ODBPPnkk1myZEl6enpSXV1dwYTQ/5x66qk5++yze639yZ/8Sf7hH/6hQomg//vrv/7rfP7zn8/HPvaxJMmYMWPy0ksvZcGCBQY2cIQaGxuT/O6n5k499dTi+q5duw75CToAOBLmeVA+ZnpQXuZ5UH7medC3ToSZ3kmVDgDHu0KhkJtvvjnf/e5389hjj2X06NGVjgT92qRJk/Lcc89ly5YtxWPcuHG57rrrsmXLFoMaKMEHPvCBPP/8873Wfvazn+WMM86oUCLo/1555ZWcdFLvv25VV1fn9ddfr1AiOH6MHj06jY2NWbduXXFt//79Wb9+fSZMmFDBZAAcL8zzoPzM9KC8zPOg/MzzoG+dCDM9T8CCPjZz5sysWrUq3//+91NXV1f83ab19fUZNGhQhdNB/1NXV5eWlpZea0OGDMmIESMOWQcOz1/91V9lwoQJaWtry7Rp0/KjH/0o999/f+6///5KR4N+66qrrsqXvvSlnH766TnnnHPy4x//OIsWLcpnPvOZSkeDfmHPnj35+c9/Xny9ffv2bNmyJcOHD8/pp5+e1tbWtLW1pbm5Oc3NzWlra8vgwYMzffr0CqYG4HhhngflZ6YH5WWeB+VnngdH7kSf6VUVCoVCpUPA8eytfl/psmXL8qlPferohoHj1MSJE3P++edn8eLFlY4C/db/+l//K/PmzcsLL7yQ0aNHZ/bs2fnP//k/VzoW9Fu7d+/Of/2v/zVr1qzJrl270tTUlI9//OP5b//tv2XgwIGVjgfHvCeeeCKXXHLJIes33HBDli9fnkKhkC9+8Yu577770tnZmfHjx+d//I//4R/vACgL8zw4Osz04MiY50F5mefBkTvRZ3oKWAAAAAAAAAAAACU66Q9vAQAAAAAAAAAA4M0oYAEAAAAAAAAAAJRIAQsAAAAAAAAAAKBEClgAAAAAAAAAAAAlUsACAAAAAAAAAAAokQIWAAAAAAAAAABAiRSwAAAAAAAAAAAASqSABQAAAAAAAAAAUCIFLAAAAAAAAAAAgBIpYAEAAAAAAAAAAJRIAQsAAAAAAAAAAKBE/w9gVXoUEBycJwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 3000x2500 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Visualizing the dataset\n",
    "plt.rcParams['figure.figsize']=(30,25)\n",
    "newdf.plot(kind='hist', bins=10, subplots=True, layout=(5,2), sharex= False, sharey=False)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "7609bae7-dcb5-4d38-a597-8f3bad3c4b07",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clas                           1.000000\n",
       "Bare Nuclei                    0.822696\n",
       "Uniformity of Cell Shape       0.818934\n",
       "Uniformity of Cell Size        0.817904\n",
       "Bland Chromatin                0.756616\n",
       "Clump Thickness                0.716001\n",
       "Normal Nucleoli                0.712244\n",
       "Marginal Adhension             0.696800\n",
       "Single Epithelial Cell Size    0.682785\n",
       "Mitoses                        0.423170\n",
       "Name: Clas, dtype: float64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "correlation=newdf.corr()\n",
    "correlation['Clas'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "60945318-f3e0-4804-be6f-6ac122b21063",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAACPQAAAg2CAYAAAB5IxtjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzdd3xN9x/H8ffNvUmEkCCSiL1r703tPWrVqlVqhao9i6ItRVFqtvYoalQVHbaW2nvPmhERe2Tce39/5JdbVxKjFfdqXs/H4z4ecu73nPP5npx7XfLO52uwWq1WAQAAAAAAAAAAAAAAAHAKLo4uAAAAAAAAAAAAAAAAAMDfCPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAABCLrVu3qk6dOgoICJDBYNAPP/zw3H22bNmiwoULK1GiRMqcObOmTZv20ucl0AMAAAAAAAAAAAAAAADE4sGDB8qfP7++/vrrFxp//vx51axZU2XLltX+/fs1cOBAdevWTcuXL3+p8xqsVqv1nxQMAAAAAAAAAAAAAAAAJBQGg0ErV65UvXr14hzTr18//fjjjzp+/LhtW6dOnXTw4EHt2LHjhc9Fhx4AAAAAAAAAAAAAAAAkGGFhYbp7967dIyws7JUce8eOHapatardtmrVqmnPnj2KiIh44eOYXkk1AAAAAAAAAAAAAAAAeK6IkHOOLiHBG/n1PA0bNsxu29ChQ/XJJ5/862MHBQXJz8/Pbpufn58iIyMVEhKi1KlTv9BxCPQAAAAAAAAAAAAAAAAgwRgwYIB69uxpt83d3f2VHd9gMNh9bbVaY93+LAR6AAAAAAAAAAAAAAAAkGC4u7u/0gDPk/z9/RUUFGS3LTg4WCaTSSlTpnzh47i86sIAAAAAAAAAAAAAAACAhKhkyZL67bff7Lb9+uuvKlKkiFxdXV/4OAR6AAAAAAAAAAAAAAAAgFjcv39fBw4c0IEDByRJ58+f14EDB3Tx4kVJUct3tWrVyja+U6dO+uuvv9SzZ08dP35cs2bN0syZM9W7d++XOq/BGr1QFwAAAAAAAAAAAAAAAOJVRMg5R5eQ4Ln6ZH7hsZs3b1aFChVibG/durXmzJmjNm3a6MKFC9q8ebPtuS1btqhHjx46evSoAgIC1K9fP3Xq1OmlaiTQAwAAAAAAAAAAAAAA8JoQ6HG8lwn0OApLbgEAAAAAAAAAAAAAAABOxOToAgAAAAAAAAAAAAAAABIMi9nRFeANQIceAAAAAAAAAAAAAAAAwIkQ6AEAAAAAAAAAAAAAAACcCIEeAAAAAAAAAAAAAAAAwIkQ6AEAAAAAAAAAAAAAAACcCIEeAAAAAMBzHTp0SO+//74yZcqkRIkSydPTU4UKFdLo0aMVGhrq6PLsbN68WQaDQZs3b37pfY8dO6ZPPvlEFy5ciPFcmzZtlDFjxn9dX3wKDQ1V06ZN5evrK4PBoHr16v3rY65du1affPJJrM99/vnn+uGHH17qeHPmzJHBYLC7xuXLl1eePHn+eZGxeFbdGTNmVJs2bV7p+eJTbPdeXNc++vru2bPnH53rWdfNYDCoa9eu/+i4r1r58uVVvnx5R5fhUAaDIc7v1bNcuHBBBoNBY8eOfe7Y2F6vAAAAAADg9SDQAwAAAAB4pm+++UaFCxfW7t271adPH/38889auXKl3n33XU2bNk3t2rVzdImvzLFjxzRs2LBYf3g9ePBgrVy58vUX9RJGjBihlStXavz48dqxY4dGjx79r4+5du1aDRs2LNbn/kmgp1atWtqxY4dSp079r2t7lmfVvXLlSg0ePDhez/8qxXbv/ZNr/yKedd3gXHbs2KEPPvjA0WUAAAAAAIB4YnJ0AQAAAAAA57Vjxw517txZVapU0Q8//CB3d3fbc1WqVFGvXr30888/v5JzPXz4UIkTJ46x3Ww2KzIy0u7cjpAlSxaHnv9FHDlyRFmyZNF7773n6FJiePTokRIlSqRUqVIpVapUDq2lYMGCDj3/y3oT7j28HlarVY8fP5aHh4dKlCjh6HIAAAAAAP+U1eLoCvAGoEMPAAAAACBOn3/+uQwGg2bMmBFroMbNzU1169a1fW2xWDR69Gi99dZbcnd3l6+vr1q1aqXLly/b7Re9xNLWrVtVqlQpJU6cWG3btrUtBTN69Gh9+umnypQpk9zd3bVp0yZJ0p49e1S3bl2lSJFCiRIlUsGCBbV06dLnzmPPnj1q2rSpMmbMKA8PD2XMmFHNmjXTX3/9ZRszZ84cvfvuu5KkChUqyGAwyGAwaM6cOZJiX/bo8ePHGjBggDJlyiQ3NzelSZNGXbp00e3bt+3GZcyYUbVr19bPP/+sQoUKycPDQ2+99ZZmzZr13NqlqKW0AgMDlSZNGrm5uSlz5swaNGiQwsLCJP29hM769et1/PhxW+3PWnZsyZIlqlq1qlKnTi0PDw/lzJlT/fv314MHD2xj2rRpo8mTJ0uS7ZjRy+8YDAY9ePBAc+fOtW2PXgIpepmeX3/9VW3btlWqVKmUOHFihYWFPXMJn23btqlEiRLy8PBQmjRpNHjwYJnNZtvzcS2nFl3Pk9+ruOqO/n48veTWxYsX1aJFC/n6+srd3V05c+bUl19+KYvFEuM8Y8eO1bhx45QpUyZ5enqqZMmS+vPPP+O81pJ09+5dmUwmjRkzxrYtJCRELi4u8vLyUmRkpG17t27dlCpVKlmtVtt8nrz3nnXto927d0+dO3eWj4+PUqZMqQYNGujq1avPrPF51y3a/PnzlTNnTiVOnFj58+fXTz/9FONYp0+fVvPmze2uZ/Sxn8disWjSpEkqUKCAPDw85O3trRIlSujHH3985n7Dhg1T8eLFlSJFCiVLlkyFChXSzJkzbdcx2saNG1W+fHmlTJlSHh4eSp8+vRo2bKiHDx/axkydOlX58+eXp6enkiZNqrfeeksDBw6M89wRERHy9fVVy5YtYzx3+/ZteXh4qGfPnpKi3jd69eqlAgUKyMvLSylSpFDJkiW1atWqGPtGL3M2bdo05cyZU+7u7po7d67tuSeX3Lpx44YCAwOVK1cueXp6ytfXVxUrVtS2bdtirdliseizzz5T+vTplShRIhUpUkQbNmyI+wI/Yf369apUqZKSJUumxIkTq3Tp0i+8LwAAAAAAeDF06AEAAAAAxMpsNmvjxo0qXLiw0qVL90L7dO7cWTNmzFDXrl1Vu3ZtXbhwQYMHD9bmzZu1b98++fj42MZeu3ZNLVq0UN++ffX555/LxeXv3zmZOHGismfPrrFjxypZsmTKli2bNm3apOrVq6t48eKaNm2avLy8tHjxYjVp0kQPHz6MEdB40oULF5QjRw41bdpUKVKk0LVr1zR16lQVLVpUx44dk4+Pj2rVqqXPP/9cAwcO1OTJk1WoUCFJcXdHsVqtqlevnjZs2KABAwaobNmyOnTokIYOHaodO3Zox44ddiGogwcPqlevXurfv7/8/Pz07bffql27dsqaNavefvvtOGt//PixKlSooLNnz2rYsGHKly+ftm3bppEjR+rAgQNas2aNUqdOrR07digwMFB37tzRwoULJUm5cuWK87inT59WzZo11b17dyVJkkQnTpzQF198oV27dmnjxo2SopZ6evDggZYtW6YdO3bY9o0+X8WKFVWhQgXb8lXJkiWzO0fbtm1Vq1YtzZ8/Xw8ePJCrq2uc9QQFBalp06bq37+/hg8frjVr1ujTTz/VrVu39PXXX8e5X2yeVXdsbty4oVKlSik8PFwjRoxQxowZ9dNPP6l37946e/aspkyZYjd+8uTJeuuttzRhwgTb+WrWrKnz58/Ly8sr1nMkS5ZMRYsW1fr169WnTx9J0oYNG+Tu7q579+5p165dKlWqlKSosETFihVlMBhiPdaLXPsPPvhAtWrV0qJFi3Tp0iX16dNHLVq0sH1v/+l1W7NmjXbv3q3hw4fL09NTo0ePVv369XXy5EllzpxZUtTSdaVKlVL69On15Zdfyt/fX7/88ou6deumkJAQDR06NM4apKhg0YIFC9SuXTsNHz5cbm5u2rdvX6whsCdduHBBHTt2VPr06SVJf/75pz788ENduXJFQ4YMsY2pVauWypYtq1mzZsnb21tXrlzRzz//rPDwcCVOnFiLFy9WYGCgPvzwQ40dO1YuLi46c+aMjh07Fue5XV1d1aJFC02bNk2TJ0+2+3589913evz4sd5//31JUlhYmEJDQ9W7d2+lSZNG4eHhWr9+vRo0aKDZs2erVatWdsf+4YcftG3bNg0ZMkT+/v7y9fWNtYbQ0FBJ0tChQ+Xv76/79+9r5cqVKl++vDZs2BAj9PX1118rQ4YMmjBhgi2MWaNGDW3ZskUlS5aMc64LFixQq1at9M4772ju3LlydXXV9OnTVa1aNf3yyy+qVKlSnPsCAAAAAIAXR6AHAAAAABCrkJAQPXz4UJkyZXqh8SdOnNCMGTMUGBioSZMm2bYXLFhQxYsX1/jx4/XZZ5/ZtoeGhur7779XxYoVbduif2CfKFEi/fLLL3YBkBo1aih37tzauHGjTKaof85Wq1ZNISEhGjhwoFq1amUXCnpSo0aN1KhRI9vXZrNZtWvXlp+fnxYtWmTriJItWzZJUUGY5y1n8+uvv+qXX37R6NGjbQGNKlWqKF26dGrSpInmzZun9u3b28aHhITojz/+sIUN3n77bW3YsEGLFi16ZqBn7ty5OnTokJYuXWrrIFSlShV5enqqX79++u2331SlShWVKFFCyZIlU3h4+AstxfPxxx/b/my1WlW6dGnlzJlT5cqV06FDh5QvXz5lyZJFfn5+khTjmCVKlJCLi4tSpUoV5/kqVaqk6dOnP7cWSbp586ZWrVpl6/hUtWpVPXr0SFOnTlXfvn1t1+1FPKvu2IwbN05XrlzRzp07VaxYMUlR95bZbNa0adPUvXt3Zc+e3TY+adKk+umnn2Q0GiVJAQEBKlasmNatW6emTZvGeZ7KlSvryy+/VFhYmNzd3bV+/XqVL19eV69e1fr161WqVCldvXpVx48fV48ePeI8zotc++rVq2vixIm2r0NDQ9W3b18FBQXJ398/1n1e5Lo9evRI69evV9KkSSVJhQoVUkBAgJYuXar+/ftLknr27KmkSZPq999/twVbqlSporCwMI0aNUrdunVT8uTJYz3+tm3bNH/+fA0aNEiffvqp3XyeZ/bs2bY/WywWlS9fXlarVV999ZUGDx4sg8GgvXv36vHjxxozZozy589vG9+8eXPbn//44w95e3vbXb8XCam8//77Gj9+vJYsWWL32p8zZ44KFy6svHnzSpK8vLzsajWbzapUqZJu3bqlCRMmxAj03L9/X4cPH47zmkXLkSOHXfjMbDarWrVqunDhgiZOnBgj0GM2m/Xbb78pUaJEkqLu+YwZM2rIkCH67bffYj3Hw4cP9dFHH6l27dpauXKlbXvNmjVVqFAhDRw4UDt37nxmnQAAAAAA4MWw5BYAAAAA4JWIXhbr6U45xYoVU86cOWMsx5I8eXK7MM+T6tataxfmOXPmjE6cOKH33ntPkhQZGWl71KxZU9euXdPJkyfjrO3+/fvq16+fsmbNKpPJJJPJJE9PTz148EDHjx//J9O1dTp5er7vvvuukiRJEmO+BQoUsAulJEqUSNmzZ7db9iuu8yRJksQukPTkef/pMjfnzp1T8+bN5e/vL6PRKFdXV5UrV06S/vE1eVrDhg1feGzSpEntlm+TokIWFotFW7dufSX1xGXjxo3KlSuXLcwTrU2bNrJarTG62tSqVcsW5pGkfPnySdJzv5eVKlXSo0ePtH37dklRnXiqVKmiypUr2wIU69evlxQV/vk3nr6WL1rj81SoUMEW5pEkPz8/+fr62o77+PFjbdiwQfXr11fixIljvFYfP378zOXJ1q1bJ0nq0qXLS9e2ceNGVa5cWV5eXrZ7esiQIbp586aCg4MlRb0O3dzc1KFDB82dO1fnzp2LcZxixYrp9u3batasmVatWqWQkJAXOn/evHlVuHBhu7DO8ePHtWvXLrVt29Zu7Pfff6/SpUvL09NTJpNJrq6umjlzZqyvvYoVKz43zBNt2rRpKlSokBIlSmQ77oYNG2I9boMGDWxhHinqNVinTh1t3brVbqm7J23fvl2hoaFq3bq13ffWYrGoevXq2r17t92yfQAAAAAA4J8j0AMAAAAAiJWPj48SJ06s8+fPv9D4mzdvSop9WaOAgADb89HiWv4otueuX78uSerdu7dcXV3tHoGBgZL0zB+6N2/eXF9//bU++OAD/fLLL9q1a5d2796tVKlS6dGjRy80v6fdvHlTJpNJqVKlsttuMBjk7+8fY74pU6aMcQx3d/fnnv/mzZvy9/ePsfySr6+vTCZTjPO8iPv376ts2bLauXOnPv30U23evFm7d+/WihUrJOkfX5OnPet7/LTozjBPiu4k80/m+DJu3rwZ530b2/mf/l5GL632vOtWqlQpJU6cWOvXr9eZM2d04cIFW6Bn586dun//vtavX6/MmTO/cGesuPzTGl/2uNHHjj7uzZs3FRkZqUmTJsV4rdasWVPSs1+rN27ckNFojLOLUFx27dqlqlWrSpK++eYb/fHHH9q9e7cGDRok6e95Z8mSRevXr5evr6+6dOmiLFmyKEuWLPrqq69sx2rZsqVmzZqlv/76Sw0bNpSvr6+KFy8eZ9eaJ7Vt21Y7duzQiRMnJEV1DXJ3d1ezZs1sY1asWKHGjRsrTZo0WrBggXbs2KHdu3erbdu2evz4cYxjvujraNy4cercubOKFy+u5cuX688//9Tu3btVvXr1WL/vsV1jf39/hYeH6/79+7GeI/q9uFGjRjG+v1988YWsVqtt6S8AAAAAAPDvsOQWAAAAACBWRqNRlSpV0rp163T58mWlTZv2meOjf9B/7dq1GGOvXr0qHx8fu21PB1Se9Vz0vgMGDFCDBg1i3SdHjhyxbr9z545++uknDR061LYkkCSFhYX9qx88p0yZUpGRkbpx44ZdqMdqtSooKEhFixb9x8d++jw7d+6U1Wq1uy7BwcGKjIyMcV1fxMaNG3X16lVt3rzZ1pVHkm7fvv0qSrZ51vf4adFBgScFBQVJ+vveiu4mEhYWZjfuRTuoxCVlypS6du1ajO1Xr16VpH90jWPj5uamMmXKaP369UqbNq38/f2VN29eZc6cWZK0efNmbdiwQbVr134l53OE5MmTy2g0qmXLlnF22XlWWClVqlQym80KCgp6qUDY4sWL5erqqp9++smu68wPP/wQY2zZsmVVtmxZmc1m7dmzR5MmTVL37t3l5+dnWzLt/fff1/vvv68HDx5o69atGjp0qGrXrq1Tp04pQ4YMcdbRrFkz9ezZU3PmzNFnn32m+fPnq169enYddhYsWKBMmTJpyZIldq+Rp+/raC/6OlqwYIHKly+vqVOn2m2/d+9erOOjX19Pb3Nzc5Onp2es+0S/FiZNmhTnsmyxhfMAAAAAAMDLo0MPAAAAACBOAwYMkNVqVfv27RUeHh7j+YiICK1evVqSbMtnLViwwG7M7t27dfz4cVWqVOkf15EjRw5ly5ZNBw8eVJEiRWJ9PLkM0JMMBoOsVqutQ0m0b7/9NsayMi/TxSR6Pk/Pd/ny5Xrw4MG/mu/T57l//36MYMK8efPs6ngZ0QGBp6/J9OnTY4x91jV5kQ5DL+revXv68ccf7bYtWrRILi4uevvttyVJGTNmlCQdOnTIbtzT+0XXJr349/LYsWPat2+f3fZ58+bJYDCoQoUKLzyP56lcubL27t2r5cuX25bVSpIkiUqUKKFJkybp6tWrL7Tc1qu89k8fV/rnnXwSJ06sChUqaP/+/cqXL1+sr9XYuvxEq1GjhiTFCKU8j8FgkMlkslsK7dGjR5o/f36c+xiNRhUvXlyTJ0+WpBjffynqe1OjRg0NGjRI4eHhOnr06DPrSJ48uerVq6d58+bpp59+UlBQUIzltgwGg9zc3OyCOkFBQVq1atULzTUuBoMhxmv60KFD2rFjR6zjV6xYYdcR6N69e1q9erXKli1rdx2fVLp0aXl7e+vYsWNxvhe7ubn9q3kAAAAAAIAodOgBAAAAAMSpZMmSmjp1qgIDA1W4cGF17txZuXPnVkREhPbv368ZM2YoT548qlOnjnLkyKEOHTpo0qRJcnFxUY0aNXThwgUNHjxY6dKlU48ePf5VLdOnT1eNGjVUrVo1tWnTRmnSpFFoaKiOHz+uffv26fvvv491v2TJkuntt9/WmDFj5OPjo4wZM2rLli2aOXOmvL297cbmyZNHkjRjxgwlTZpUiRIlUqZMmWINIFSpUkXVqlVTv379dPfuXZUuXVqHDh3S0KFDVbBgQbVs2fJfzTdaq1atNHnyZLVu3VoXLlxQ3rx59fvvv+vzzz9XzZo1Xyj88bRSpUopefLk6tSpk4YOHSpXV1ctXLhQBw8ejDE2b968kqQvvvhCNWrUkNFoVL58+eTm5qa8efNq8+bNWr16tVKnTq2kSZPG2SnpeVKmTKnOnTvr4sWLyp49u9auXatvvvlGnTt3Vvr06SVFLQdUuXJljRw5UsmTJ1eGDBm0YcMG21JhL1r303r06KF58+apVq1aGj58uDJkyKA1a9ZoypQp6ty5s7Jnz/6P5hSbSpUqyWw2a8OGDZo7d65te+XKlTV06FAZDAZbOO5ZXuW1f/q40otdt7h89dVXKlOmjMqWLavOnTsrY8aMunfvns6cOaPVq1dr48aNce5btmxZtWzZUp9++qmuX7+u2rVry93dXfv371fixIn14YcfxrpfrVq1NG7cODVv3lwdOnTQzZs3NXbs2BgBl2nTpmnjxo2qVauW0qdPr8ePH2vWrFmSZHsttW/fXh4eHipdurRSp06toKAgjRw5Ul5eXi/Ueatt27ZasmSJunbtqrRp08Z4jdauXVsrVqxQYGCgGjVqpEuXLmnEiBFKnTq1Tp8+/dzjx6V27doaMWKEhg4dqnLlyunkyZMaPny4MmXKpMjIyBjjjUajqlSpop49e8piseiLL77Q3bt3NWzYsDjP4enpqUmTJql169YKDQ1Vo0aN5Ovrqxs3bujgwYO6cePGS4exAAAAACBBslgcXQHeAAR6AAAAAADP1L59exUrVkzjx4/XF198oaCgILm6uip79uxq3ry5unbtahs7depUZcmSRTNnztTkyZPl5eWl6tWra+TIkc/syvEiKlSooF27dumzzz5T9+7ddevWLaVMmVK5cuVS48aNn7nvokWL9NFHH6lv376KjIxU6dKl9dtvv6lWrVp24zJlyqQJEyboq6++Uvny5WU2mzV79my1adMmxjENBoN++OEHffLJJ5o9e7Y+++wz+fj4qGXLlvr8889jBAn+qUSJEmnTpk0aNGiQxowZoxs3bihNmjTq3bu3hg4d+o+OmTJlSq1Zs0a9evVSixYtlCRJEr3zzjtasmSJChUqZDe2efPm+uOPPzRlyhQNHz5cVqtV58+fV8aMGfXVV1+pS5cuatq0qR4+fKhy5cpp8+bN/6gmf39/TZ48Wb1799bhw4eVIkUKDRw4MEa4YP78+frwww/Vr18/mc1m1alTR999952KFCnywnU/LVWqVNq+fbsGDBigAQMG6O7du8qcObNGjx6tnj17/qP5xKVgwYLy8fFRSEiIXdAjOtBTsGDBF3qtvMpr/6SXuW5xyZUrl/bt26cRI0bo448/VnBwsLy9vZUtWzbVrFnzufvPmTNHhQoV0syZMzVnzhx5eHgoV65cGjhwYJz7VKxYUbNmzdIXX3yhOnXqKE2aNGrfvr18fX3Vrl0727gCBQro119/1dChQxUUFCRPT0/lyZNHP/74o6pWrSopKlQ0Z84cLV26VLdu3ZKPj4/KlCmjefPm2S2vF5fKlSsrXbp0unTpkgYNGiQXF/sG2e+//76Cg4M1bdo0zZo1S5kzZ1b//v11+fLlZ4ZpnmfQoEF6+PChZs6cqdGjRytXrlyaNm2aVq5cGeu90bVrVz1+/FjdunVTcHCwcufOrTVr1qh06dLPPE+LFi2UPn16jR49Wh07dtS9e/fk6+urAgUKxPpeCQAAAAAA/hmD1Wq1OroIAAAAAAAAAAAAAACAhCDi2nFHl5DguabO6egSnsvl+UMAAAAAAAAAAAAAAAAAvC4EegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCIEegAAAAAAAAAAAAAAAAAnYnJ0AQAAAAAAAAAAAAAAAAmF1WpxdAl4A9ChBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ8KSW/hPiAg55+gSgHi3NfcAR5cAxDs3g9nRJQDx7mv3SEeXAMS7VAZ3R5cAxLv8ka6OLgGId0FGq6NLAOLdPQNLHeC/b2LQH44uAYh3J3Nld3QJQLzLsG+9o0sA8JrRoQcAAAAAAAAAAAAAAABwIgR6AAAAAAAAAAAAAAAAACdCoAcAAAAAAAAAAAAAAABwIgR6AAAAAAAAAAAAAAAAACdicnQBAAAAAAAAAAAAAAAACYbF4ugK8AagQw8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE7E5OgCAAAAAAAAAAAAAAAAEgyrxdEV4A1Ahx4AAAAAAAAAAAAAAADAiRDoAQAAAAAAAAAAAAAAAJwIgR4AAAAAAAAAAAAAAADAiRDoAQAAAAAAAAAAAAAAAJwIgR4AAAAAAAAAAAAAAADAiRDoAQAAAAAAAAAAAAAAAJwIgR4AAAAAAAAAAAAAAADAiZgcXQAAAAAAAAAAAAAAAECCYTE7ugK8AejQAwAAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAACAEyHQAwAAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAACAEyHQAwAAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAACAEzE5ugAAAAAAAAAAAAAAAIAEw2pxdAV4A9ChBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ0KgBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ0KgBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ0KgBwAAAAAAAAAAAAAAAHAiJkcXAAAAAAAAAAAAAAAAkGBYLI6uAG8AOvQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBETI4uAAAAAAAAAAAAAAAAIKGwWi2OLgFvADr0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE7E5OgCAAAAAAAAAAAAAAAAEgyLxdEV4A1Ahx4AAAAAAAAAAAAAAADAiRDoAQAAAAAAAAAAAAAAAJwIgR4AAAAAAAAAAAAAAADAiRDoAQAAAAAAAAAAAAAAAJwIgR4AAAAAAAAAAAAAAADAiRDoAQAAAAAAAAAAAAAAAJyIydEFAAAAAAAAAAAAAAAAJBhWi6MrwBuADj0AAAAAAAAAAAAAAACAEyHQA4eIjIx0dAkAAAAAAAAAAAAAAABOiUAPXiuLJap1mMkUtdrbxo0bdfr0aUeWBAAAAAAAAAAAAAAA4FQI9OC1cnGJuuVWr16tLFmyaNCgQdq5c6fCwsIcXBkAAAAAAAAAAAAAAIBzMDm6APz3Wa1WGQwG29eTJ0/W559/rh49eui9995TokSJ5O7uHmMcAAAAAAAAAAAAAABAQkSgB/HGbDbLaDTaQjqRkZEymUzau3evatasqd69e8tisSgyMlJhYWGyWCzy8PCQxWKxdfIBAAAAAAAAAAAAAABIaAj0IN4YjUZJ0vTp07V//3516NBBefPmVUhIiFxdXTV27FidPXtWN2/e1L59+1S2bFnNnj2bMA8AAAAAAAAAAAAAAEjQSE4g3uzfv1/58uXTqFGjlCFDBkVGRsrV1VWdO3dW4sSJNW3aNHl4eKhgwYLq3r27Vq5cqW+++cbRZQMAAAAAAAAAAAAAEH8sZh6OfrwB6NCDV8JqtdqW1oo2ZswY5cuXTwsWLLDbXqNGDZUrV06JEye2LcMVHBysOXPmyMfH53WWDQAAAAAAAAAAAAAA4HQI9OBfMZvNMhqNMcI8586d0/Hjx9W+fXtJ0h9//CGLxaLLly+rcOHCyp49uywWi+7evavbt29r6NChslqtKliwoCOmAQAAAAAAAAAAAAAA4DQI9OBfMRqNkqRFixbp0KFDSpMmjdq0aaPMmTMrICBAs2fP1siRI5UrVy5dv35dISEhCggI0Lp167Rt2zZ9//332rhxo/Lly6dVq1Ypbdq0Dp4RAAAAAAAAAAAAAACAYxHowb9y6dIltWjRQmfOnFG9evU0fvx4/frrr+rTp4+WLl2qJUuWyGg0KmPGjMqWLZtOnDihtm3b6uDBgypVqpSuXr2qDz/8UCVKlJAkWSwWubi4OHhWAAAAAAAAAAAAAAAAjkOgB//KsmXLlChRIp05c0YeHh46evSoSpUqJQ8PD82ZM0dt27a1G799+3b5+fkpV65c8vX1VWBgoCTJarXKYrHYOv4AAAAAAAAAAAAAAAAkVLRCwXOZzeZYt92/f19r165Vp06d5OHhoSFDhujtt99WhQoVNGLECCVOnFiStGXLFq1du1bt2rXT+++/r3r16snPz09Wq1VSVJjHYDAQ5gEAAAAAAAAAAAAAABAdevAMTwdtNm/erDt37qhChQpKliyZPD099ddff2nJkiUaOHCgjEaj5s6dq9q1a0uSrly5ooCAAAUFBWn8+PHy9fXVrl27lDNnTrvzGAyG1z43AAAAAAAAAAAAAAAcwmpxdAV4AxDoQZyigza3bt1S48aNdeDAARmNRhUoUEC9e/dW5cqV1aZNG3388ccaM2aMevXqZdv3yJEjmj9/vtq1a6fGjRurWLFiypQpk6So7j4uLi4EeQAAAAAAAAAAAAAAAGLBkluw8+TyWpGRkZo5c6YmTpyo3Llz6/Tp0/ruu+9kNBr15Zdf6u7du3rnnXeUMWNG7d+/X0FBQXr06JGOHj2qfv36af/+/TKZTDIYDHZhHqPRSJgHAAAAAAAAAAAAAAAgDgR6ErgLFy7YfR29vJbZbNbevXs1fvx4TZ48WaVLl5a3t7cqVKig1q1bKzQ0VJMnT1bu3Ln17bffavPmzSpatKjq1q2r4sWLy8vLS8uXL1fmzJljPT4AAAAAAAAAAAAAAABiR6AngYqMjFTbtm3VvXt3XblyRVJUiOfSpUsqWLCgtm3bpuLFi6t58+YKDw+X1Wq17VuzZk2VLFlSK1eu1OHDh1WxYkX9+uuvmjVrlho2bKg///xTixYtUtKkSe06/gAAAAAAAAAAAAAAAOD5CPQkQFarVSaTSYULF9bVq1e1Zs0aSVHdc/bv3y9vb2+VK1dOktS4cWOVLFlS33//ve7fvy9J8vT0VP369eXt7a1x48ZJknLlyqUqVaqoU6dOypMnjywWiywWCx15AAAAAAAAAAAAAAAAXhKBngQoumtOly5dlCZNGq1Zs0aHDh2SJC1btkxFihSRwWCQ1WpV1qxZVadOHV26dEkLFiywHaNcuXIqXLiwjhw5onPnztkd32q1ysXFRS4u3F4AAAAAAAAAAAAAAAAvi8RFAmI2m23deaJ16tRJFy9e1KpVq3Tt2jVt2LBBtWrVkiSFh4dLkpo0aaKsWbPqxx9/1F9//WXb98MPP9SGDRuUOXNmu/MYDIbXMBsAAAAAAAAAAAAAAID/JgI9CYTVapXRaJTBYNCOHTs0ePBg3b59W9WqVVOJEiX0559/avz48cqVK5cKFCggSXJ3d5ckpUyZUvXr19fVq1c1YcIE2zEDAgKULFkyWSwWB8wIAAAAAAAAAAAAAIA3kMXCw9GPNwCBnv+gq1evasuWLXbbDAaDbt26pQYNGqhWrVo6c+aMDhw4IEnq1q2bbty4ocWLF9s69LRo0UKLFi3ShQsXJEkNGjRQ3bp1VaNGjRjnY2ktAAAAAAAAAAAAAACAV8f0/CF4k5jNZn366acKDw9XuXLl7J4bP368goODtWfPHqVPn9629FbOnDnVsGFDrVq1Sq1bt1b58uU1f/589e/fX1arVenSpdOKFSs0fPhwR0wJAAAAAAAAAAAAAAAgQSHQ8x9jNBr15ZdfysPDw2779evXNX36dA0ZMkSZM2eW2WyWJEVERMjV1VUdO3bUb7/9posXLypnzpyaM2eO7t+/r5MnT8rX11f+/v6SopbuMhgMr31eAAAAAAAAAAAAAAAACQVrJf0HeXh4KCQkRNWqVdPVq1clSeHh4UqePLmSJUsm6e9lslxdXSVJ3t7eateunXbu3Kk5c+ZIkjw9PVW4cGGlS5dOlv+vIUeYBwAAAAAAAAAAAAAAIH4R6HlDhYeHKzIy0vbnp0VEROj48ePq1auXJMlkMilRokQ6fPiwbt68KYPBYNv/wIEDCgsLU7NmzVSyZEkVKlQoxvGiA0AAAAAAAAAAAAAAAACIX6Q03iBWq1WStHv3bmXMmFG7du2SJLm5uUmSDh06pODgYEmSn5+fJk6cqCVLlmjbtm1KnTq13nnnHa1fv14rV66UFBXyCQoK0tChQ7Vu3TpJ0qxZs1S9evXXPTUAAAAAAAAAAAAAAAD8n8nRBeD5rFarLBaLjEajJKlo0aJydXXV9OnTVbhwYW3fvl0tW7aUm5ubXFxc9M0336hChQqqXr26ateurY8++kj79u3TwIEDdeHCBQ0ePFiLFy9W1qxZtXz5chUpUkTFihWTFLWklsVioSNPArLnwGHNXrRMx06c0Y2bofpq5GBVervUM/fZvf+Qxkz6RmfO/yVfn5R6v3kjNalfy27Mb5t+16Rv5+nSlWtKlya1unVorcrlSsfnVIBnStOmqjJ0qSM3X289OHlZpwfP1e2dJ+Icb3AzKVOvhvJvWFbuvt56fO2mLkxYqWvfbY4x1q9eKeWZ/pFurNutQ23GxuMsgGdL3aaa0gbWlZtvcj04eUnnhszR3Z3H4xxvcDMpfc935dvobbml8lbYtZu69NUKXf9uoyTJt0l55fiqa4z9fs/QTNawiHibB/AsVVvWUJ2O9eSdKrkun76kucNm6sTuY8/dL0eRtzR0yWe6dPKi+tXsYdtuNBlVL7Ch3m5UUSn8UujauStaOGqeDm7ZH5/TAJ6pbIuqqtSxjrx8vXXt1GUtHz5XZ3fH/rklW4lc+mjx0BjbR1Tqoetno5ZgLtW0ooo1eFsBOdJJki4ePq/VY77TXwfPxt8kgOfI2aqy8neqKQ9fb906dUV/frJAQbtOxjo2dcmcqv39oBjbl5broztnr0mSDCajCnSto+yNyiqxf3LdOXdNuz5fosubD8XrPIBnKdyyskp2rKWkqbx14/QV/TJsvi7tjv0+z1Aip1ot+TjG9ikVe+vm/+9zSXJPllgV+jTWW9WLyCNZEt2+fEO/fbpQZzYdjLd5AM9SskUVletYW0l9vXX91GX9OHyeLsRxn2cukVOdFg+JsX1MpV668f/PLU/KX6ek3pvUTUd+3a15Hca98tqBF9WhQ0v17NlR/v6+OnbstPr0GaY//tgV69hSpYrqs88GKHv2LEqc2EMXL17Wt98u1KRJM21jWrZspG++iXlPe3llU1hYWLzNA3gWz3fryqvVuzL6pFT4uQu6NXaKwvYfiXVsyk/6yLNutRjbw89e0LV3P4g6Xv2aSlK7ilyzZIx67vhp3f56psKPxv53BJAgWS2OrgBvAAI9Ts5sNstoNMpoNOrIkSPq3LmzBgwYoNmzZ6tatWpq3LixFi5cqL59+6p48eIaPXq02rRpo2XLlqlo0aIaPHiwypYtq2+++Ubt27fX119/rU2bNmnjxo26evWqpk2bpoYNG9qdkzBPwvLo0WPlyJpZ9WpWVY9Bnz53/OWrQQrsPUQN61TXyCF9tP/QMX365WSl8PZSlQplJEkHjhxX76Ej1fWDVqpUrpQ2bNmu3oNHat7UscqX+634nhIQg+87JZV9RGud7D9Tt3edVJpWlZX/uwH6s2xPhV25Ges+eb/pLrdU3jrec7oenQ+Sm08yGf4frHxSorQ+yjq0hW7tiDs0AbwOPu+UUubhbXSm/7e6u/uEUresojyLBmrv2z0UdiUk1n1yzugl11ReOt1jih5dCJKrj1eM+zzy7gPtKf2R3TbCPHCUkrVLq/WQtpo5eLpO7jmhys2racDcwepZ+UPdvBr7fS5JHkkTK3Bcdx3545C8fLztnmvS+z2VrV9O0/tP0dUzl5W/XEH1ntFfgxv014Wj5+N5RkBMhWqXVMMhrbVk8Eyd23NSZd6rrMA5A/RplZ66dTX2zy2SNLxCdz26/9D29f2bd21/zlYit/b+uF3f7zupyLAIVe5YV13mD9JnVXrpzvVb8TofIDaZ6xRXyU9a6I9Bc3R99ym91aKiqs/vo+8r9NODZ9znS8v2Vvj9R7avHz9xnxft20hZG5TWtr4zdfvMVaUtl09Vvu2uH98ZpptH/4rX+QCxyVW7hKoNaam1g2fr8p5TKtS8oprP7auplfvq7jPu88nleynsifv84RP3uYurUS0W9NeDm3e1rPNE3bsWqmSpUyj8weN4nQsQl/y1S6jOkFb6YfAsXdhzUsXfq6x2c/rryyq9dfsZ9/noCj30+In7/MET93k07zQ+qjXwPZ17xi+pAK9Do0Z1NHbsUH300cfavn2PPvjgPa1aNVcFC1bSpUsxg2gPHjzU1KlzdPjwCT18+FClShXV11+P1MOHjzRz5iLbuDt37ipfvgp2+xLmgaMkrlpeKXp3VujIiXp88KiSNqwl30kjdbVRO5mDgmOMDx07RbcmfWv72mA0KvXiGXq4fqttW6LC+fXg500KO3hU1vBwebVuIr8pX0Qd80bcf0cAAOyR3HBSFktUIs9oNCosLEzvvfee8uXLp9y5c6ts2bKqWLGiypQpo2bNmikyMlJdunRR8eLFtXz5cplMJn3zzTe6deuWihYtqs6dO2vYsGEKDw9X0qRJVbduXU2YMEFLly61hXnMZrMjpwsHKluyqLp1aK0q5V+se87SH9bI389X/bt3UpaM6dWobnXVr1VVc75bbhszf8kPKlm0kNq3aqLMGdKpfasmKl6kgOYv/SGeZgE8W/pOtXR10UZdXbhRD09f0enBcxV25abStqka6/gUFfLLu2QuHWg+Ure2HtbjSzd0d/9Z3dlzyn6gi0G5p3yoc2O+16O/rr+GmQBxS9Oxjq5/t1HXF23Qo9NXdG7IHIVduanUrWO/z5NXKCCvkrl09L3PdXvbYYVduqH7+8/o3p6nfkvGKkXcuG33AByl1gfvaOOS9dq4eL2unLmsucNn6ua1EFVt8ewlYzt83ll/rNqqU/ti/hZY2QbltXLyMh3YtFfBl67rtwU/6+CWA6rd/p34mgbwTBU/qKUdSzdqx5KNun72ipYPn6tb126qbIvY38+j3bt5R/du/P2wWqy25+Z2n6RtC37VlWN/6frZq1rUf7oMBoNylM4b39MBYpW3Qw2dXLxZJ7/brNtnrurPTxbo/tWbytWq0jP3e3Tzrh7duGN7PHmfZ21QRgcm/ahLGw/q3sUbOj5/gy5vPqS8HWvG93SAWJX4oIb2L9msA4s3K+TMVf06fIHuXrupIi0qP3O/Bzfv6sGNO7bHk/d5gcbllcjbU0vbj9flPad050qILu05pevHL8b3dIBYlf2glnYv3aRdSzYp+OxVrR4+T7ev3VSJFlWeud/9m3d1/8Yd2+PJ+1ySDC4GNZvQRb+NX6bQSzF/kAy8Tt26faA5c5Zo9uzFOnnyjPr0GabLl6+qQ4eWsY4/ePColi79UcePn9Jff13Wd9+t1G+/bVHp0sXsxlmtVl2/fsPuAThKsvca6v4PP+v+D+sUef6ibo2dKvP1YCVtVCfW8db7D2S5ecv2cMuVXS7JPHX/x59tY0I+Hqn73/+oiFNnFXnhkm6OGCcZDEpUrNDrmhYA/CcQ6HFCVqvV1iVnwoQJCggI0KFDh3T+/HlNmzZNSZMmlSTNmTNH9+/fV6JEiWQwGGz7f/bZZ1q2bJm2bo1Kwnbv3l03btxQ3759Y5zryeAQ8CIOHjmhUk994CpdvJCOnjitiMjIqDFHj6tU0afGFCusA4f5jRq8fgZXo5Lmy6zQp1rth245KK8i2WPdJ1W1Irp38JwydK2r0gemquT28co6tIVcErnajcvUq5HCb97VtUWb4q1+4EUYXE1Kmi+zbm22b7N/a8tBJSuaI9Z9UlQronsHzyptl3dUbP90Ff5jojINbSWXRG5244xJEqnonqkqtm+6cs0foCR5MsXbPIBnMbqalDlvFh3adsBu+8GtB5S9cNwdAMu/W1F+Gfy1bMLiWJ93dTMp4qmuU+GPw5SjSK5/XTPwsoyuRqXLk1nHt9l/bjm+7aAyFY79c0u0fmu+0Ge7punDhR8rW8nczxzr5uEuo6tJD2/f/9c1Ay/LxdUon7yZdGWrffv+K1uPyK9Itmfu2+DnT/Xe3q9Vc/EApS6V0+45o7tJ5qfezyMfR8i/6LNfO0B8cHE1KnXeTDq37bDd9rNbDytt4Wff5+3Xfqbuu79Wi0UDlKGk/eeR7FUK6cq+06oxoo167Jmijr+OUukudWVwMcRxNCD+GF2NSpMnk0499bnl9LZDyviczy3d14zUx7umqP3CQcpSMubn7sofNdSD0HvavXTzqywZeGmurq4qVCiv1j/RdUSS1q/fphIlCr/QMfLnz60SJQpr27Y/7bZ7eibRqVPbdebMTq1YMVv58z/7MzwQb0wmueXMrkd/7rHb/GjHXrnnf7H/G/GsV0OPd+6T+VrcIUxDInfJZJLlbsyubACAuLHklhMyGAxas2aNOnfuLEkKCAiQt7e3MmTIIIvFIhcXF5nNZmXIkEFdunTRqlWrdOrUKb31VtQPMpo2baoZM2ZoxowZKlCggDJkyKDFixcrS5YsMc7F8lp4WSGht5QyubfdtpQpkivSbNbt23eVyieFQm7eUsoUT4/xVkho6OsrFPg/1xTJ5GIyKvzGHbvtYTfuKIWvd6z7eGTwlVexHLKEhevw+2PlmiKpcoxqJ9fknjrefZokyatoDgU0r6BdlfrF9xSA53JNkVSGWO7ziBt35JrKO9Z9EqX3k1ext2QJi9CxtmPkmiKpso5qL5O3p073mCJJenT6ik5+9LUeHr8oY9LEStO+pvL/+Kn2Veqlx+eD4ntagJ1kyZPKaDLqTshtu+13Qu7IO1XyWPfxz5hazfq10ifvDpTFHPua1Ae3HlCtD+rq+M6juv5XkPKUzqciVYvzORkO4Zk8mYwmo+499X5+78YdJXtqubhod4JvaVH/6bp4+Lxc3U0qWv9tfbjwY33VdLjO7oo9UP9Ov+a6ExSqE38cjvV5ID4lSpFULiajHj51nz+6cUcecXxueXj9trb2/VYhhy7I6GZStoZlVGvxAP307mcK2hnVfe3ylsPK276Gru08obsXgpWmTG5lrFZIBt7P4QCJk0fd5w9C7O/zByF35JnKK9Z97gff1k/9vtW1I+dldDMpX4MyarlogOY1+UwXd52QJCVP5yvvkrl0eNV2fddmtFJm8lf1EW3kYjRq28SV8T4v4ElJ/v+55X4sn1uS+sR+n98Lvq1l/b/RlcPnZHR3VaH6ZdV+4SBNbzpC5/9/n2conF1FG5fXhJoD4n0OwPP4+KSQyWRScLD9Es/BwTfk55fqmfueObNTqVJF7f/pp+M1e/bfv2Ry8uRZtW/fS0eOnFCyZEnVpUtbbdq0QkWLVtPZsxfiYypAnIzeXjKYjLLctF+O2Rx6S8aUKZ6/v08KeZQqppBBnz9zXPJuH8h8I0SPdu77V/UCQEJDoMcJrVixQs2aNdPw4cPVt29fHT16VEWKFNG8efPUqlUrWSwWW0eeCRMmaO7cuZozZ46GDh0qDw8PSdLYsWNVpEgRbdu2TRkyZFD9+vUlRXX/ebKbz5soLCwsxlqyLmFhcnd3d1BFCc/T95DVav3/9mePedPvPbzZrHqqfbPBIFmtsQ92MUhW6UjnSTLfi1rT/fTQ+co7s4dO9p8pg9Go3FO66kSvGYoIvRffpQMv7ul72iApjtvc4OIiq9Wqk4FfyXzvoSTp3NC5yvltL50d8K0sj8N1b99p3dt32rbP3V0nVPC30QpoV1PnPp4VT5MAni3GbW74+7OI3XYXF3Wb2FPfj/9O185fjfN4cz75Vh1HddH4jV/LapWu/xWkzd9vUPl3n73sCxC/Yn5uefqzTLTgc9cUfO6a7evz+04reeqUqty+dqyBnsod66pw3dL6qukwRT7VzQR4rWL93BL7fX7n3DXdeeI+D953RkkCUihfx1q2QM+OIfNVdnQ7vbt5jGS16u5fwTq5ZKtyNHk7vmYAPNfTn1EMBkOc/wy9ee6abj5xn1/Zd0bJUqdUyQ41bYEeg4tBD27e1Zr+38pqsSroyAV5+iVXyY61CPTAYZ6+pZ/1ueXGuWu68cR9fnHfaXmnTqFy7Wvr/K4Tck+SSM0mdNHyAd/o4S3+vwXOI/b38zje0P+vcuVGSpIksYoXL6QRI/rr7NkLWrr0R0nSrl37tWvXftvY7dt3688/1yow8H316jX01U8AeAEx3ruf9f/nT0hSp5os9+7r4aY/4hyTrHVjJa5WQdc79JLC+XcoALwMAj1OqEGDBgoODpaXV9RvMmTLlk2dO3dWv3791Lx5c5lMUd+2yMhImUwmjRo1Sv369VO9evVUokQJWa1WFSpUSKtXr1atWrXsjv1fCFSMHDlSw4YNs9v2cZ9uGtL3IwdVlLD4pEiukFD7pHbordsyGY3y8koWNSZlcoXcfHrMHaVMHvtvzwPxKSL0riyRZrk/9du+bj7JYnQziRZ+/bbCgkJtYR5JenD6igwuLnJPnVLGxO7ySO+rfPP/XsowusV5hSuL9GepHnr01/VXPxkgDhGh92SNNMvtqa5Trj5einiqm0m08OBbCg8KtYV5JOnh6csyuLjILXWK2DvwWK26d+CsPDKnfoXVAy/m7q17Mkea5f3U+3mylF4xuvZIkoenh7Lkz6aMuTOr7fAOkqLeq11cXLTo7HJ91vITHd1+WPdC72psh5FydXeVp3dS3boequb9Wyn4Eu/jeP3u37orc6RZSZ+6zz19kuleSOyfW2JzYf9pFa1fNsb2Su1rq2qXevr6vU919cTFf1su8I88Dr0nS6RZiZ/63OLh46VHL3GfB+87o6wNStsd97cPJsjo7ir35J56GHRLxQY20b2LN15V6cALe3gr6j73fOr9PHHKZDG69jzLlf1nlLf+3/f5/eDbMkeaZbX8/cO1kDNXldQ3uVxcjbJEmP917cCLemD73GLfjcfTJ5nuh7z4cioX959RwfplJEkpMvgpRTpftfm2j+356P9vGXlmgcZU7KnQi3Ev5wK8aiEhoYqMjIzRjSdVKp8YXXueduHCJUnS0aMn5evro48/7mEL9DzNarVq795Dypo14yupG3gZ5tt3ZI00x+jGY0zuLfNTPwuKjec71fRg7XopMjLW55O1fFdebZvreqe+ijh9/pXUDAAJCX2HnZSXl5ct4e3u7q6uXbvKxcVFQ4YMkSRZLBZbsCcwMFDp06fXsGHDFBoaagvtRId5npcUf9MMGDBAd+7csXv0+6iTo8tKMPLneUs7dtu3RNy+a59yv5VNrv+/J/PnzhlzzO59KpA352urE4hmjTDr3qFzSlEun932FG/n0509p2Ld5/buk3L3Sy5j4r87fyXOklpWs0Vh127q4Zmr+rNcb+2q1M/2CPllr279cVS7KvXT46vP/gc98KpZIyJ179A5eT91nycvl093d5+MdZ+7u07IzS+FXBInsm3zyBwgq9ms8GtxL5HomSejwq8//x/zwKtmjojUucNnla9sAbvt+coW0Km9J2KMf3TvoXpX6aZ+NXrYHusX/qIrZy6rX40eOrPf/u+AiLAI3boeKqPJqOI1SmrPr7viczpArMwRZl06ck5vlbF/P3+rTD6d3xv755bYpM2dUXeC7d+rK3Woo+ofNtSU1iN18fC5V1Iv8E9YIswKOXxeacrmsduepmweXd9zOo69YkqZJ6MeBd+Osd0cFqGHQbdkMBmVsWYxXfiVlv54/SwRZl07fF6Zn7rPM5fNq8t7X/w+98+dUfeeuM8v7TmlFBn87Fokp8zkr3vXbxHmwWtnjjDrypHzyvbU55ZsZfLqwkt8bgl44j6/cfaqvqzaRxNq9rc9jq3fq7M7jmlCzf66c+3mq5wC8FwRERHat++wKlWyD8tXqlRWf/6594WPYzAY5O7u9swx+fLlUlAQgTU4QGSkwo+fkkfxwnabE5UorLCDx565q3vh/HJNn1b3f1gX6/PJWjWW1wctdL3rAIUff/G/GwAAf6NDjxN7sptO5syZNWDAAPXq1UudOnVS+vTpZbVaZTabZTKZNG7cOA0ZMiTWDjz/ha48T3J3d4+xvFZEOD88/6cePnyki5f/XobiytXrOnHqrLySJVVqf1+NnzpbwSE3NXJwb0lS43q19N3y1Ro9cYYa1q2ug0eOa8VPv2rMJ/1sx2jR+B216dJHMxcsVYWyJbVp2w79uXu/5k0d+9rnB0jSxWlrlPvrrrp78Kzu7DmtNC0ryT2tj67M/U2SlGVQM7n7p9CxDydLkq4v/12ZejRQzq8CdX7MUrmmSKZsQ97T1e82yfI4qiXogxOX7M4RcedBrNuB1+XK9NXKMelD3T94Tnf3nFTqFlXknsZH1+b9KknKOLC53FKn1KkPJ0mSglf8rvQ9Gin7V110ccwSmVIkVaYhLRX03SZZHodLktL3eld3957S43PXZEyaWAEf1FSS3Bl1ZsC3DpsnErY1365S1/HddfbQGZ3ed1KVmlWVT4CPflv4iySpWd8WSuGfUpN7fiWr1apLp+w7kNwJuaOIsAi77VkLZFMK/5S6cPS8UvinVKMeTWVwMejH6SxbAcfY+O0atRrXVRcPndX5fadVunklpQjw0baFUZ9b6vZtJi+/FJrfK+pzS/m2NRV6OVjXTl2W0dWkYvXLqmDNEvqm45e2Y1buWFe1ejbW3I8m6ublYNtv0oc9eKzwh2ExiwDi2eEZ61T+q866ceicgvee0VvvVZBnmpQ6Pn+DJKlo/8ZK4p9cm7tPlyTlaVdN9y6H6Nb/7/OsDUorc61i+q39BNsxUxXMoiT+yXXz6F9K4p9ChXo2kMFg0KGpPzliioD+/Had6o3vrKuHzuvKvtMq2KyivAJSau/CqPu8Yt8mSuqfXKt6TpMkFWtbXXcu39CN/9/neeuXUc6axfR9x/G2Y+5dsF5F21RVtU9aavecX5Uik79Kd3lHu+f84pA5Atu+XaMm47ro8qFzurjvlIo3ryTvAB/9uXC9JKl636by8kuuJb2mSpLKtK2hW5dv6Pr/7/OC9csoX83imtdxnCQpMixC109dtjvH47tRXWWf3g68LhMnfqtZs8Zr375D+vPPfWrXrrnSpQvQN98skCSNGNFPAQH+ateuhySpY8dWunTpqk6ePCNJKl26qLp376CpU+fYjjloUHft3LlPZ89eUNKknurS5X3lz59L3bt//NrnB0jS3YXL5TOin8KOn1LYoWNK2qCWTP6+urd8tSTJu2s7GX19dHPIF3b7edarrrDDxxVx9kKMYyZr3VjendsoZOBIRV4NkkvKqBUcrA8fyfrocbzPCXgjWCyOrgBvAAI9DhK9XNaLMhgMatq0qebNm6fu3btrxYoVkiSTySSr1apq1aqpWrVq8VUu/sOOnDitth/+HcYZPWmGJOmdGpX12ce9FHIzVNeu//2bAWkD/DVl7HCNnjhD361YLV+flBrQvZOqVChjG1Mwby6NGdZfk2bM06Rv5itdmtQaM3yA8uV+6/VNDHhC8Kodck2eVJl6NpS7X3LdP3FJB5uP0uPLUWFAN19vJUqT0jbe/DBM+xt/puyfv6+iv4xUxK17uv7jnzo3arGjpgA8V8iq7XJNnlTpezaSm29yPThxUUfe+1xh0fe5X3K5p/Gxjbc8fKzDTYYry2ftVODnLxR5655urN6uv564z03Jkijb2E5yS+WtyHsP9eDweR2qN0T395957fMDJGnHT38oafJkatitiZL7JtelUxc1qs0IhVyJWk7F2zeFUgakes5R7Lm6u6lJ7/fkm85Pjx8+1oFNezW5+3g9vPsgPqYAPNe+n3YoiXdS1fiooZKlSq5rpy5pyvujdOtK1Pt5Ml9vpXjic4vJ1aT6A1vKyz+FIh6HR41vM1LHNh+wjSnbsopc3V31wbRedudaO+F7rZ2w7LXMC3jSudU75Z48qQp1r6/Evt4KPXlZP7cao/tXojovJPb1VpInPre4uJlUfHBzJfFPrsjH4bp98op+bjVGlzYetI0xuruqSJ93lTR9KkU+DNOljQe0+aOpCr/7MMb5gdfh2E9/yiO5p97uVl+evt66ceqyvmszRnf+/37u6eutZAF/v58bXU2qPKi5kvqnUOTj8P+PH60zm/6+z+9eC9XClqNUdXBLdfx5pO5ev6Vds3/W9qmrX/v8AEk6+NOfSuydVJU/aqBkqbwVdOqSZr3/hW4/8bnF+4n3c6OrSbUGvmf73HL91GXNavOFTjzxuQVwNsuWrVaKFN4aOPAj+fv76ujRU6pXr7UuXrwiSfL391W6dAG28S4uLhoxop8yZkynyMhInTv3lz7+eJS+/XahbYyXVzJNmTJKfn6pdOfOPR08eFSVK7+rPXsOxjg/8Do8/HWzQr2Sybt9Cxl9Uij87AUFdxso87Wonw0ZfVLI5O9rt4/BM4kSVyyrW2OnxHrMpO/WlcHNTanGDrXbfnv6PN2ZPi9+JgIA/0EG639tPSYnZ7Va7TrmbNu2TQEBAcqSJcsL7b927VrVqVNHa9eujTXA87JBof+KiBBaxuO/b2vuAY4uAYh3bgbaxOO/72v32NcUB/5LUhncnz8IeMPlj3R1dAlAvAsy8t+G+O+7Z+A3o/HfNzHoD0eXAMS7k7myO7oEIN5l2Lfe0SXgFQo7RKdNR3PP5/wNU1wcXUBCYrFYbGGeX375RalSpVKbNm1UpEgRDR8+XJcuRS3TYjbH/cPMcuXKqXr16lq3Lvb1KBNimAcAAAAAAAAAAAAAAOC/hPTHa+Ti4qIzZ87o1KlTmjVrlgYPHqz69etr/vz5Wr16tQ4ePKjly5fLaDTGeYwkSZJoyZIl8vT0fI2VAwAAAAAAAAAAAAAA4HWhQ088iq3TTvfu3dWgQQOFhITogw8+ULp06TRw4EB16dJFu3fv1sqVKyVFdfOJS3SY51mdfAAAAAAAAAAAAAAAAPBmItATD6zWqPXFozvtPHr0yPbc+PHjlTJlSoWHh8vF5e/LX758eZUoUULff/+9JNk9F5dndfIBAAAAAAAAAAAAAADAm4lAzysUHeQxGAySpB9++EFlypRRkyZNtG7dOj148EDZsmVTq1atFBwcrC1bttj2TZs2rQwGg63rTvSxAAAAAAAAAAAAAADAf4fVaubh4MebgEDPKxIZGWkL8kjSzp079cknn6hq1ap68OCBBgwYoHnz5kmSBg0aJJPJpHnz5unSpUu2fR48eCAfHx9JsjsWAAAAAAAAAAAAAAAAEg6Towt405nNZhmNRplMJoWHh2vmzJkqWLCgfv/9dzVv3lx9+/bV/fv31a1bNy1btkwlSpRQwYIF1bNnT3300Uc6ceKEmjZtqu3bt2vLli368ccfHT0lAAAAAAAAAAAAAAAAOBAdev4lo9EoSdq/f798fHz05ZdfqkaNGvrqq69UrVo1SZKnp6eaNWumyMhIzZkzR5LUoUMHlS9fXsHBwQoJCVHu3Ll15swZVahQwVFTAQAAAAAAAAAAAAAAgBMg0POSrFar3denT59Ww4YNtWnTJo0fP15nzpzRF198IUmaOXOmbVyVKlVUqlQp7d27V2vXrpUk9ezZU0ajUVmyZNGnn34qPz8/RUREvL7JAAAAAAAAAAAAAAAAwOkQ6HlJBoPB7uvz58/r999/1+eff66SJUtKklq3bq127dppw4YNOnDggG3se++9Jw8PD02aNElhYWGqUqWK3n77bS1atEi7du2SJLm6ur62uQAAAAAAAAAAAAAAAMD5EOj5ByZMmKDu3btLkooVK6YuXbrozp078vf3lyS5u7urdu3aSps2rUaNGmXbL0+ePKpdu7YaNWokk8kkSRo+fLh27typtWvXKjw8/LXPBQAAAAAAAAAAAAAAAM6FQE8sHjx4YPuzxWKJ8bzJZNLEiRN17NgxeXt7q169esqRI4d69+5tG1OkSBHVq1dPhw8f1uLFi23bu3Xrpnbt2sloNMpsNitjxoxasGCBunbtKjc3t/idGAAAAAAAAAAAAAAAAJwegZ4nWK1WVa5cWZ988okeP34sSXJxcdHRo0ftxrVr106lS5dW3759JUlvvfWWunbtqmXLlmnfvn22cVWqVFGGDBm0ceNG27boJbusVqtcXKIuf8OGDeXj4xOvcwMAAAAAAAAAAAAAAE7AauHh6McbIMEHeqxWq6SoTjwGg0Fly5bVrFmzdOTIEUnStWvXVKBAAQ0bNsw23sPDQyNGjNC6dev0008/yc3NTdWqVVOZMmVsIR9Jypo1q6ZMmaIZM2bEOK/BYLCFewAAAAAAAAAAAAAAAIBoCTbQEx3kiQ7VREZGSpKGDh2qZMmS6auvvtK9e/eUOnVqDRkyRBMmTFBoaKhtfLly5dSsWTP1799fZrNZmTJlUseOHbVx40YtWrTIdp6MGTNKksxm82ucHQAAAAAAAAAAAAAAAN5UCTLQY7VabcGclStXqlGjRurQoYOtk87kyZO1cOFCbd26VZLUoUMH+fv7q1evXpL+7ubTokULHTt2TLNmzZIklShRQl9//bXefvvtGOc0Go2vY2oAAAAAAAAAAAAAAAB4wyXIQI/BYNC5c+dUtmxZde3aVcWKFVOePHnk5+cnSapZs6YqVqyokSNH6vr16/Lz89PgwYM1b9487du3Ty4uUZft/Pnzcnd3V8eOHRUaGio/Pz8FBgYqbdq0tg5AAAAAAAAAAAAAAAAAwMswOboAR7h37566d++udOnSacmSJQoICLA9ZzabZTQaNWPGDGXNmlUrVqxQp06dVL9+fdWuXVstW7bU+PHjlTp1am3atEk//PCDbt++rRQpUtiO8WQHIAAAAAAAAAAAAAAAAOBlJMgOPatXr9bWrVsVGBio1KlT27ZbrVYZjUaZzWZlzpxZgYGBGjVqlE6dOiUPDw/NnDlTJpNJXbt2VdGiReXp6akKFSqoSZMmdscnzAMAAAAAAAAAAAAAAIB/KkF26Nm9e7cCAgJUpkwZu+1PB3EmTpyoBQsWaPbs2Ro0aJB8fHy0YcMGXbp0SUmSJFH27NlfZ9kAAAAAAAAAAAAAAOBNZ7E4ugK8ARJkh56//vpLiRIl0tWrV2M8F92lR5JcXFw0cuRIjR07Vnv27JEk+fj4qGDBgsqePbssFossvNAAAAAAAAAAAAAAAADwCiXIQE/lypV19OhRnTp1yrbNarVKiurSEx4ermHDhkmSOnfurFy5cunOnTsxjuPi4iIXlwR5CQEAAAAAAAAAAAAAABBPEmQapUGDBgoICNCUKVNsXXqeXG5ry5Yt+vXXX3X27FlJ0v79+1WvXj1HlAoAAAAAAAAAAAAAAIAExuToAhzB399fQ4YMUbt27eTt7a3Ro0fL3d1dkZGRWrFihSZOnKj69esrU6ZMkiSj0SiLxSKDwWAX/AEAAAAAAAAAAAAAAABetQQZ6JGk999/X6dOndKsWbM0b9485c2bVwaDQefOndOYMWP0/vvv241naS0AAAAAAAAAAAAAAAC8Dgk20CNJn3/+uT788EOtXr1akZGRcnV1VYcOHWzPWywWgjwAAAAAAAAAAAAAAAB4rRJ0oEeSAgIC1LFjR7ttkZGRMplMhHkAAAAAAAAAAAAAAADw2iXoQI/BYIixzWq1ymRK0JcFAAAAAAAAAAAAAADEF6vF0RXgDUALmqfEFvIBAAAAAAAAAAAAAAAAXhcCPQAAAAAAAAAAAAAAAIATIdADAAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAIATIdADAAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAIATMTm6AAAAAAAAAAAAAAAAgATDYnZ0BXgD0KEHAAAAAAAAAAAAAAAAcCIEegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCIEegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCIEegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCImRxcAAAAAAAAAAAAAAACQYFgtjq4AbwA69AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4ERMji4AAAAAAAAAAAAAAAAgwbBYHF0B3gB06AEAAAAAAAAAAAAAAACcCIEeAAAAAAAAAAAAAAAAwIkQ6AEAAAAAAAAAAAAAAACcCIEeAAAAAAAAAAAAAAAAwIkQ6AEAAAAAAAAAAAAAAACcCIEeAAAAAAAAAAAAAAAAwIkQ6AEAAAAAAAAAAAAAAACciMnRBQAAAAAAAAAAAAAAACQYVoujK8AbgA49AAAAAAAAAAAAAAAAgBMh0AMAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAgBMh0AMAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAgBMh0AMAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAgBMxOboAAAAAAAAAAAAAAACABMNicXQFeAPQoQcAAAAAAAAAAAAAAABwIgR6AAAAAAAAAAAAAAAAACdCoAcAAAAAAAAAAAAAAABwIgR6AAAAAAAAAAAAAAAAACdCoAcAAAAAAAAAAAAAAABwIgR6AAAAAAAAAAAAAAAAACdicnQBAAAAAAAAAAAAAAAACYbF4ugK8AagQw8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAAAAE6EQA8AAAAAAAAAAAAAAADgREyOLgAAAAAAAAAAAAAAACChsFrNji4BbwA69AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4ERMji4AAAAAAAAAAAAAAAAgwbBYHF0B3gAEevCfsDX3AEeXAMS7t4+OdHQJQLyL3Pmjo0sA4t3n/dY5ugQg3nlnuOXoEoB4F3wqiaNLAOKdX64Hji4BiHcPg4yOLgGIdwfMuRxdAhDvfGolc3QJAAC8ciy5BQAAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAACAEyHQAwAAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAACAEyHQAwAAAAAAAAAAAAAAADgRAj0AAAAAAAAAAAAAAACAEzE5ugAAAAAAAAAAAAAAAIAEw2pxdAV4A9ChBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ0KgBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ0KgBwAAAAAAAAAAAAAAAHAiBHoAAAAAAAAAAAAAAAAAJ2JydAEAAAAAAAAAAAAAAAAJhsXi6ArwBqBDDwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAAAToRADwAAAAAAAAAAAAAAAOBETI4uAAAAAAAAAAAAAAAAIMGwWhxdAd4AdOgBAAAAAAAAAAAAAAAAnAiBHgAAAAAAAAAAAAAAAMCJEOgBAAAAAAAAAAAAAAAAnAiBHgAAAAAAAAAAAAAAAMCJEOgBAAAAAAAAAAAAAAAAnAiBHgAAAAAAAAAAAAAAAMCJmBxdAAAAAAAAAAAAAAAAQIJhsTi6ArwB6NADAAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAIATIdADAAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAIATIdADAAAAAAAAAAAAAAAAOBECPQAAAAAAAAAAAAAAAIATIdADAAAAAAAAAAAAAAAAOBGTowsAAAAAAAAAAAAAAABIMKwWR1eANwAdegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCIEegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCIEegAAAAAAAAAAAAAAAAAnQqAHAAAAAAAAAAAAAAAAcCImRxcAAAAAAAAAAAAAAACQYFgsjq4AbwA69AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9AAAAAAAAAAAAAAAAABOhEAPAAAAAAAAAAAAAAAA4EQI9MAhzGazo0sAAAAAAAAAAAAAAABwSiZHF4CExWq1ymq1ymg0SpKuXbum1KlTO7gqAAAAAAAAAAAAAABeE4vF0RXgDUCHHrxWBoNBLi4u2r9/vypWrKi6deuqcuXKWrJkiaNLAwAAAAAAAAAAAAAAcAp06EG8s1gscnH5Ozv2448/qnPnzmrWrJmqVaumI0eOqF27dnr48KFatWpl694DAAAAAAAAAAAAAACQEBHoQbwxm80yGo12YR6r1aq1a9eqWbNmGjt2rCTp6tWrevjwoW7cuKGIiAgCPQAAAAAAAAAAAAAAIEFjyS28Elar1fZni8ViC/NI0urVq9W4cWNJUUtubdy4UbVq1dKOHTuUKVMmffHFF1q2bJn69u2rRIkSOaR+AAAAAAAAAAAAAAAAZ0GHHvxrJ06ckIeHhzJkyGAX5Dl27Jhat26tvXv3SpJ27typAgUKqECBAmrYsKFcXFzUp08fdenSRZ6enrp//762bNmiKlWqyM3NzZFTAgAAAAAAAAAAAAAAcBg69OBfCQ4OVmBgoJYvXy5JMhqNioiIUIsWLZQ3b16VK1dO33//vXLnzq3ixYvL3d1d+fPnV4oUKTRq1Cj169dPSZIkkST9/vvv+uqrr3Tx4kVHTgkAAAAAAAAAAAAAAMCh6NCDlxK9tJbBYJAkpUqVSitXrpSXl5dtzPLly3XlyhUdOHBAefPm1fTp05U0aVKFh4fLzc1NDRo00IEDB/Tpp5/K1dVV6dOn15o1azRv3jx16tRJ6dOnd8jcAAAAAAAAAAAAAACId1aLoyvAG4BAD16YxWKRi0tUU6cTJ04oceLESp8+vby8vHTy5EkNHDhQy5cvV9OmTdW0aVPbfocPH5aPj4/c3NxktVqVM2dOjR49WsOGDdOYMWPk4uIiDw8PrVq1SiVLlnTU9AAAAAAAAAAAAAAAAJwCgR68MBcXF128eFE9evTQ8ePHVbp0aX3zzTeSpIsXL2rdunX68ssv1atXL0VGRspkMslqtWr37t1q3LixpL87+2TKlEmzZ8+W2WzWhQsXlDVrVklRoaHocwEAAAAAAAAAAAAAACREpCbwwsaNG6f8+fPL3d1d3377rV0XnhIlSqh379764osvdOfOHZlMJkVEROjRo0d68OCBMmTIIEm6fv266tSpo02bNslgMMhkMtnCPGazWS4uLoR5AAAAAAAAAAAAAABAgkZyAi/k3LlzWrFihb766istWrRIpUqVUqVKlWzPJ02aVA0aNFBAQIB69eolSXJ1ddX9+/d19+5dZcqUSePGjVPWrFkVFBSkt956K8Y5jEbja5sPAAAAAAAAAAAAAACAsyLQgxeyefNmHTx4UE2aNLFtCwkJ0Z07d3T9+nVJUu7cudWxY0ctWbJE+/fvlyRt3bpVly9fVvny5TVhwgQtW7ZMu3fvVurUqR0yDwAAAAAAAAAAAAAAAGdHoAc2VqtVUtTSV09LmzatkidProULF+rcuXPq16+fOnbsqFKlSilLliwaMWKEbt++rcaNG6tUqVLq3bu3pKiluLJly6axY8fq4sWLqlatmqxWa6znAAAAAAAAAAAAAAAAgGRydAFwDosXL9bixYv1ww8/xLr0Va5cudSoUSMNGjRI169fV8mSJVWuXDlVqVJFwcHBmjZtmlKkSKEuXbooMDBQ9evX13fffadmzZrpxIkTMhgMkqTIyEiZTCaW1wIAAAAAAAAAAAAAAIgDgR7IbDbr0aNHWrdunX7++WdVr17dFryJljZtWo0bN04tW7aUp6enMmTIILPZLA8PD0nSsmXLdOLECUlRXXkmTZqksmXLSpIMBoPMZrOMRqPdMQEAAAAAAAAAAAAASHAsFkdXgDcAS24lUI8ePVLNmjW1YMECGY1GVa1aVY0bN1afPn0kSSaTybYEl/T3clwFCxZUtmzZ5ObmZgvzXLlyRe7u7sqaNaskyc/PT126dFHatGlt+9GRBwAAAAAAAAAAAAAAvImmTJmiTJkyKVGiRCpcuLC2bdv2zPGTJ09Wzpw55eHhoRw5cmjevHkvfU4CPQmUi4uLvL299fHHH0uS0qRJozZt2ujmzZsaP368JMnyRCowesmsaNFdfQ4dOqQOHTrI1dVVderUsRtjtVpj7AcAAAAAAAAAAAAAAPCmWLJkibp3765BgwZp//79Klu2rGrUqKGLFy/GOn7q1KkaMGCAPvnkEx09elTDhg1Tly5dtHr16pc6L4GeBMrd3V1Dhw7VgwcPNGzYMElSoUKF1Lp1a40aNUr37t2T0Wi0C/VEO3z4sHr16qVWrVrp7bffVvLkyfXrr78qc+bMduMI8wAAAAAAAAAAAAAAgDfZuHHj1K5dO33wwQfKmTOnJkyYoHTp0mnq1Kmxjp8/f746duyoJk2aKHPmzGratKnatWunL7744qXOS6AnAXlyCS1JypYtm/r166dRo0YpKChIyZMn17vvvis/Pz/16tUrzuO4ubkpW7Zsyp49u/744w8tWLBAnp6eMpvN8T0FAAAAAAAAAAAAAACAfyUsLEx37961e4SFhcUYFx4err1796pq1ap226tWrart27fHeexEiRLZbfPw8NCuXbsUERHxwjUS6EkgIiMjbR1zorvuuLi4qEWLFsqaNat69OghScqZM6c6deqkxYsX6/Dhw3JxcYkR1MmRI4c6dOigzz77TLlz55bFYpHFYpHRaHy9kwIAAAAAAAAAAAAAAHhJI0eOlJeXl91j5MiRMcaFhITIbDbLz8/Pbrufn5+CgoJiPXa1atX07bffau/evbJardqzZ49mzZqliIgIhYSEvHCNBHr+Y57uwhPNZDIpMjJSAwcO1MCBAzVx4kTdunVL/v7+GjJkiJYsWaI///xTHh4eqlq1qkqWLGkL+cQW1HF1dZUUFQ5ycXGRiwu3EgAAAAAAAAAAAAAAcH4DBgzQnTt37B4DBgyIc3x0A5VoVqs1xrZogwcPVo0aNVSiRAm5urrqnXfeUZs2bSTFnr+ICymM/wir1RrrDRMd8NmwYYPSpUunTZs2SZKmTJmijh076siRI3r33XdVo0YNffjhh5KkLFmyqFOnTjpw4IAWLlwo6e+uPk8jyAMAAAAAAAAAAAAAwEuwWng4+OHu7q5kyZLZPdzd3WN8q3x8fGQ0GmN04wkODo7RtSeah4eHZs2apYcPH+rChQu6ePGiMmbMqKRJk8rHx+eFbxPSGP8BFotFBoNBBoNBu3fv1vz587Vx40a7ZbZmz56t999/Xzt27NCoUaMUGBio5cuXa/ny5ZKkIUOG6MiRI5o7d64MBoOKFy+uChUqaPXq1ZII7gAAAAAAAAAAAAAAgITFzc1NhQsX1m+//Wa3/bffflOpUqWeua+rq6vSpk0ro9GoxYsXq3bt2i+VvTD9o4rhVFxcXHT16lVbx50SJUroxo0bypIli9KlS6fz58/r+PHjGj58uK5cuaLAwEBt2bJFQ4cOVffu3SVJhQsXVmBgoHr06KHmzZsrICBAU6ZMUapUqRw7OQAAAAAAAAAAAAAAAAfp2bOnWrZsqSJFiqhkyZKaMWOGLl68qE6dOkmKWr7rypUrmjdvniTp1KlT2rVrl4oXL65bt25p3LhxtgYrL4NAzxsseomtP/74Qx06dFDWrFm1adMm+fn5KVGiRLbuPO7u7jp+/LiGDRumVatWqUqVKtqzZ4+yZs0qSTpy5Ijy5Mmjzp07a9GiRdq8ebOqVKliC/OYzeaXWscNAAAAAAAAAAAAAADgv6BJkya6efOmhg8frmvXrilPnjxau3atMmTIIEm6du2aLl68aBtvNpv15Zdf6uTJk3J1dVWFChW0fft2ZcyY8aXOS6DnDRYd2Jk1a5YyZ86smTNn2q23ZrFYJElp06ZV/fr1tXDhQm3YsEHlypWzjfn555+1ZcsWffTRR8qaNavOnj2rxIkT252HMA8AAAAAAAAAAAAAAEioAgMDFRgYGOtzc+bMsfs6Z86c2r9//78+54svzgWndPbsWS1dulTvvfeeXZhHilqKK3r9tQ8++ECStG/fPh07dkwPHz7UkiVL1KtXLz1+/Fienp6SpMSJE8tischqtb7eiQAAAAAAAAAAAAAAAEASgZ43wo0bN/T777/H+tyjR48UEREhV1dXSVJkZKTd82azWZJUoUIFTZw4UVOnTlXlypVVuXJldejQQR07dtT48eNtgR4pKggU3f0HAAAAAAAAAAAAAAAArxeBHic3a9Ys+fn5adasWbE+HxkZqRQpUmjHjh2yWq0ymexXUYteLuvy5csKDAzUzz//rPnz56tbt24KDg5Wt27dJP29PBcAAAAAAAAAAAAAAAAcy/T8IXCUsLAwTZ48WWPGjFGvXr1s2y0Wi20prQIFCih79uz66aefVL9+fZUuXVpms9kW5ImIiFC7du2UI0cODRo0SJkzZ1bmzJltx4qMjJTJZLIdDwAAAAAAAAAAAAAAxCMabuAFkOJwUlarVTdu3FCSJEl048YNnTt3ToGBgXr06JEtfBO9nNbo0aN17tw5TZw4UdevX7eFeSRp7dq1unbtmsqUKRPrOZ7u6AMAAAAAAAAAAAAAAADHItDjpAwGg9KmTauSJUtq9uzZypo1q5InT243xmg0ymKxqFixYhoyZIi2bNmiqlWrqnfv3po8ebIqVaqktm3bqn79+ipXrlys5wAAAAAAAAAAAAAAAIBzoT2LE3lyKS2z2azjx49r9uzZMpvNqlmzpj777DNJUZ11ng7j9OvXTxUqVNCYMWP0+++/69ChQ8qWLZtWrFghLy+vOPcDAAAAAAAAAAAAAACAcyHQ4wSsVqssFottqaywsDC5u7srW7ZsWrdunXbs2KHZs2dr8eLFatq0qd3Y6ACQq6urSpcurdKlS0uS7ty5YwvymM1mGY1GwjwAAAAAAAAAAAAAAABvAAI9TsBgMMhoNOrs2bP65JNPZDQaVaxYMVWvXl2FCxeWp6enfv/9d82ePVu1a9eWp6dnnN12rFarJMnLy0tWq1VWq9UW/gEAAAAAAAAAAAAAAIDzc3F0AYgyb9485c+fX5GRkTIajRo/frwaN26s+/fvK0eOHKpZs6Zu3bqlKVOmPPM4BoPBFvQxGAy2Dj4AAAAAAAAAAAAAAAB4M5D2cAIPHjzQd999pz59+ui7777TzJkztXDhQj1+/FidOnWSJNWqVUtFixbVjz/+qDNnzshgMMhisTi4cgAAAAAAAAAAAAAA8FKsFh6OfrwBCPQ4AYvFoj179ih9+vS2bYUKFdKgQYO0fPlyHT9+XClTplS9evUUFhamTz/9VJLovgMAAAAAAAAAAAAAAPAfRCLkNbBarc98PjQ0VFmzZtW1a9dsXXdMJpOKFCmit956S3v27JEkVaxYUY0bN1bjxo3jvWYAAAAAAAAAAAAAAAA4hsnRBSQEBoNBkrR8+XKVKlVKqVOntns+Q4YMypIli3bs2KF9+/apSJEikqI695w5c0YBAQGSJKPRqD59+rze4p1QWFiYwsLC7LaFW81yMxgdVBEAAAAAAAAAAAAAAMCrQ4eeeBBbR57Nmzfr3XfflYeHh912s9ksSerTp48uXbqkzz//XNu3b1dwcLDmzJmjPHnyKGvWrM89fkIycuRIeXl52T2+e3Dc0WUBAAAAAAAAAAAAAAC8EgR6/qWnwzVWq9XWkedJuXLlUqZMmbR582a77UajUVarVfnz59fw4cN1//591a9fX8WLF9eSJUs0cuRIZciQwW6f2I6fkAwYMEB37tyxezRLktPRZQEAAAAAAAAAAAAAALwSLLn1LxkMBkVERMjV1dX29aNHjzRr1iyVLl1aBQoUkCQ9evRI/v7+evToUZzHqlu3rqpWraqjR48qNDRUVapUeR1TeOO4u7vL3d3dbhvLbQEAAAAAAAAAAAAAgP8KOvT8C2azWevXr1f58uVt21avXq358+dr6dKleuedd3T48GFZLBb9j717j/96vv8/fn9/3h+FolBKDiGnnDKNyWEapQOhpYgJOTfH5TCMOX0d4usQoznOac6smcMIIadFmDBbCH2lHKYlqT6f9/v3R78+W8sONvV+f+p6vVzel/V5f17v1/vxwsVlu1xuezzbt2+f2bNnZ+zYsQ2fnedvN+4svfTS6dy5c0PMU1dXt2geBgAAAAAAAACAqiDo+ZpeffXVbLjhhpkxY0aKxWJKpVJmzZqV/v37p2XLljnrrLPSp0+fjBo1KhtttFEOP/zwXHzxxUmS3r175/XXX09dXV2KxX9vo0xtrSVKAAAAAAAAALDYKJW8Kv1qBNQiX1O5XM6JJ56YZs2aJUnWX3/9TJ48OePGjcuPf/zjnHPOOQ3X3njjjbntttsydOjQtGjRIpMnT07Lli1TX1+fYrE432YeAAAAAAAAAABIBD1f26abbppNN900s2bNyrRp01IsFjNw4MC88MIL+eCDDxquK5fLadWqVY444ojU1NTkwQcfzFNPPZVZs2blyy+/TNOmTSv4FAAAAAAAAAAAVCtHbv0bSn+3bmn69OnZa6+90rNnz6y22mq58MIL069fvzz33HO57777Gj5TLpeTJEOGDMnJJ5+czp07p76+PmPHjl3kzwAAAAAAAAAAQOMg6PkX6uvrU1Mz/1+m5s2bZ+DAgZk0aVLuvPPOJMmOO+6YzTbbLBdddFGSLHCkVufOnXPttddmmWWWaQiE/j4UAgAAAAAAAAAAQc+/UCwWM3HixBx99NG5/vrr8+abb6ZQKOS73/1u+vTpk5NPPjlJsuGGG6ZPnz757LPPcuGFFyZJXnrppTz++ONJkrq6urRr1y6rr756nn/++SRZIBQCAAAAAAAAAABFyd+pq6ub7+fHHnssXbp0yQsvvJBhw4ald+/e+fzzz9O2bdvsv//+mTVrVs4666wkSbdu3dKzZ8+cccYZ6dWrVzp37px33303SVJbW5s33ngjn3/+eTbffPNF/lwAAAAAAAAAADQOgp6/U1tbm2Tudp3PPvsskyZNyvnnn5+nn346I0eOTLFYzJFHHpkk2XzzzTN48OBceumlmTp1atq0aZPjjjsul112WTbffPO899572X///ZPMDYWOOuqotG3bNttvv32lHg8AAAAAAAAAqKRSyavSr0agttIDVINSqdRw/NXrr7+eAQMGZMqUKWnbtm0+//zz/PKXv0ySrLfeernwwgvz/e9/P/vvv3+233779O/fPw8++GCGDBmSu+66KyuttFJDxJPMDXlqampSW1ubO+64IyussEIlHhEAAAAAAAAAgEZiid7QM+94rZqamvz5z3/Oww8/nCuuuCJ77LFHRo4cmW7duuXdd9/Nxx9/3PCZXXfdNT169Mgpp5ySurq6rL/++tlnn30ybty4+a5L5oZCtbW1DbGQmAcAAAAAAAAAgH9liQx66uvrk/z1eK3x48fnyiuvTM+ePfP000/nkEMOydZbb52LL744m266aa677rp88sknDZ8fNmxYxo4dm5///Oepra3NfvvtlwkTJqRVq1bzfc+8kAcAAAAAAAAAAP5dS2RxUiwWkySTJk3KCiuskBNPPDGHHXZYttxyy8yZM6ch+EmSESNGZOTIkXn00UdT+v/nqG200UbZb7/98sorr6RcLqdFixapqalp2PgDAAAAAAAAAAD/qSUy6Jk6dWoGDBiQO++8M8cff3zuuuuurLjiijnmmGPy2Wef5cknn2y4dquttkq/fv0ybNiwvPvuuw3vX3HFFbnqqqtSKBQa3pu38QcAAAAAAAAAAP5Ti33Q87fbduaZOHFiXnnllZxwwgnp1KlTlllmmSTJXnvtlQ033DC333573nzzzYbrr7zyyowbNy4jR45s2NIzL975qvsDAAAAAAAAAMB/arEMekqlUkN4M+94ralTpzbENxtttFGOOeaY1NfXp1WrVkmSmTNnJknOPPPMvPjii3n44Ycze/bsJEmrVq1y//3359BDD01Nzfx/yebdHwAAAAAAAAAAvgmLTdBz9913p3379vnggw9SU1PTEN7ce++92XjjjbPzzjunW7dueeaZZ9KsWbPstttu2WabbfLjH/84SbLMMsukVCplq622yq677poRI0bk+eefb7h/r169Gq4BAAAAAAAAAPiPlMtelX41AotF0HPrrbfmqKOOymGHHZZ27do1bOIZMWJEDj/88AwePDhHHnlkmjdvnj333DP33HNP2rVrl2OPPTYvvfRS7r333iTJnDlzkiSnnnpqWrdundatWy/wXX+/oQcAAAAAAAAAAL5Ji0WdMmnSpLRs2TInnXRSSqVSPv744yTJb3/72+y888750Y9+lEGDBuW+++7LJptskptuuikTJ05M9+7ds9tuu+XUU09NkjRt2jRz5sxJu3btMnr06GywwQaVfCwAAAAAAAAAAJZAjTLoufHGG3PFFVc0/NyhQ4cs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FIvFBX630047pW/fvrn33nvTrFmzhve33XbbdOnSJQ899FAmTpyYww8/PN/+9rfzyiuv5O23387hhx+e7t27J0lKpVJqapwGBwAAAAAAAAAsmQQ9fC3ztvIUi8X85S9/ySWXXJJ27dpl/fXXz3bbbZckOffcc3Pvvfdm7Nix2XzzzbP00kunWCymc+fOGT58eJo3b54k2WKLLbLFFlvMd//6+vqvDIUAAAAAAAAAAJYU1qDwtcw7Ymv48OFZc80188ADD+TGG29M9+7dc/fdd2fGjBlZb731cuCBB+aiiy7KCy+80PDZTz75JG3atMns2bMXuG+pVEoSMQ8AAAAAAAAAsMSzoYcFzNvC8/d/nmfUqFG56667cuWVV2bPPfdMkvTs2TMXXHBBVlhhheywww65/PLLs/LKK2fvvffOAQcckGbNmuXMM8/MYYcdljZt2izwnY7YAgAAAAAAAACYS0XBfCZOnJgPPvggSfLll18uEPMkyUorrZRTTjkle+65Z/7whz+kd+/eef755zNhwoTcc889+fDDD9O0adP87//+byZNmpRp06bllVdeyfXXX58LL7zQFh4AAAAAAAAAgH9C0EODTz75JIMGDUq/fv2SJEsvvXRmzZqVSy65JCNHjsw777yTJNlggw3So0ePPP744+nXr19WXnnlTJkyJWeccUZuuOGGPPPMM0mSgw46KGuvvXY+//zzXHnllenfv3/K5XLq6+sr9owAAAAAAAAAUFHlklelX42AoGcJVi6XUy6XG35eYYUV8sMf/jBvvfVW7r///rzzzjtp06ZNrrnmmhx55JEZOHBgpk6dmmWWWSZJcu+992aDDTbI5ZdfniZNmmS55ZbLrFmz8otf/CKvvfZakmTEiBG58cYbM3r06NTV1aVQKNjQAwAAAAAAAADwTwh6llClUimFQiGFQiF/+MMf8u6776ampibbbbddevXqlRNOOCH33Xdfhg0blvHjx+fqq69OoVDIIYcckiT5/PPP8+STT2adddZJkyZNkiTjx4/PAQcckKWXXjpt2rRJknTr1i1bbbVVTjjhhHz22WeVelwAAAAAAAAAgEajttIDUBk1NTV57733cuyxx+aNN97I1ltvnWuuuSbt2rXL/vvvn7Fjx+aMM87ImDFjkiQ77bRTZsyYkT322COPPvpodtxxx+yyyy65/PLL8+abb+b999/PnDlz8uijj6Z169ZJkrq6utTW1ua2227L008/nVatWlXykQEAAAAAAAAAGgUbepZQF110UTp16pSmTZvmmmuuycCBAxt+17lz5/Tv3z/Tp09P27ZtkySFQiHf+973sscee2TIkCFJkrPPPjvnnXdeVlhhhfTu3Tu///3vG2Ke+vr61NbWplwup127dunfv/+if0gAAAAAAAAAgEZI0LMEevvtt3PPPffk0ksvzS9/+ctsvfXW2XHHHRt+36JFi+yxxx5Zd911c9xxxzW8v8IKK+TYY4/NlClTMmzYsCTJYYcdlmuuuSZnnXVWkrlbeZKkWCwmmRsCAQAAAAAAAADw7xP0LIFGjx6dV155JXvuuWfDex9//HGmTZuWKVOmJEk6duyYI444InfeeWdeeumlhus6deqUvffeO/fff3/De8ViMeVyOeVyObW1TnEDAAAAAAAAAPhvCHoWU+VyOcnco6/+3mqrrZYVVlght9xyS95+++2ceOKJOfTQQ7P11lunQ4cOOeuss/LZZ59lwIAB6dKlS0444YSGzy677LI577zz8sQTT8x3z0KhYBsPAAAAAAAAAMA3wDqVxdBtt92W2267Lb/61a8ajr76WxtuuGH22GOPnHLKKZkyZUq6dOmS7bffPt27d8/UqVMzYsSIrLjiivnhD3+YIUOGpG/fvrntttuy1157JUmWX375JHOP17KRBwAAAAAAAAC+hlK50hPQCKgxFjP19fWZOXNmHnzwwTz00EPp2bPnAuHNaqutlosuuij77rtvmjdvnvbt26e+vj7LLLNMkuSuu+7KH/7whyTJVlttlcsuuyzbbrvtAt8l5gEAAAAAAAAA+OY5cmsxMHPmzPTu3Ts333xzisVidtpppwwYMCDHH398krnhzbwjuJK/Hsf1rW99K+uuu26aNGnSEPP83//9X5o2bZp11lknSdKmTZv88Ic/zGqrrTbfPQAAAAAAAAAAWDgEPYuBmpqatGzZMj/5yU+SJKuuumr233//fPLJJ7n44ouTJKVSqeH6QqEw3+fnbfX5/e9/n0MOOSRLLbVU+vTpM9815XJ5gc8BAAAAAAAAAPDNE/QsBpo2bZqf/vSnmTFjRs4444wkyeabb5799tsv5513XqZPn55isThf1DPPq6++mqFDh2bQoEH57ne/mxVWWCEPP/xw1l577fmuE/MAAAAAAAAAACwagp5G6u+Pv1p33XVz4okn5rzzzsuHH36YFVZYIf3790+bNm0ydOjQf3ifJk2aZN111816662Xp59+OjfffHOaN2+e+vr6hf0IAAAAAAAAAAB8BUFPI1RXV9ewMWfe1p2ampr84Ac/yDrrrJNjjz02SdKxY8ccdthhue222/Lqq6+mpqZmgVBn/fXXzyGHHJL/+Z//yUYbbZRSqZRSqZRisbhoHwoAAAAAAAAAgCRJbaUH4B8rl8tfedRVbW1t6urqctppp6VUKqVdu3bZd99907Zt25x22mnZc889c/TRR2errbbKTjvtlJEjR+bYY4/NqFGjvjLUWWqppZLMjYNqajReAAAAAAAAALCwlP//4g74Z9QbVahcLn9lzDPvmK1HH300q6++eh5//PEkyRVXXJFDDz0048ePT//+/dOrV68ceeSRSZIOHTrksMMOy8svv5xbbrklyV+3+vw9MQ8AAAAAAAAAQOUpOKpMqVRKoVBIoVDI2LFjc9NNN+Wxxx6b75it66+/PgcccECeffbZnHfeeRkyZEjuvvvu3H333UmS0047LePHj88NN9yQQqGQ73znO/ne976X++67L4lwBwAAAAAAAACgmjlyq8rU1NTkgw8+aNi4s9VWW+Wjjz5Khw4dsvrqq+edd97JG2+8kTPPPDP/93//lyFDhuSJJ57IT3/60xxzzDFJks6dO2fIkCE59thjs/fee6ddu3a54oor0rp168o+HAAAAAAAAAAA/5JVLVVi3nFaTz/9dLp3756ampo8/vjjue666/LII4+kffv2qampSdOmTfPGG2/kjDPOyEYbbZQmTZrkhRdeyGmnnZbll18+48ePT21tbQ4//PA0bdo0o0ePTpKGmKe+vr5SjwgAAAAAAAAAwL/Bhp4qMe84reuuuy5rr712rr322rRq1arh96VSKUmy2mqrpW/fvrnlllvy6KOPZvvtt2+45qGHHsoTTzyRo48+Ouuss07eeuutLLvssvN9T7FYXARPAwAAAAAAAADAf8qGniry1ltv5Y477sg+++wzX8yTzD2Kq6Zm7t+ugw46KEkybty4vP766/niiy9y++23Z+jQofnyyy/TvHnzJMmyyy6bUqnUsP0HAAAAAAAAAIDqJ+hZxD766KOMGTPmK383c+bMzJkzJ0sttVSSpK6ubr7fzzsu63vf+16GDx+eK6+8Mt26dUu3bt1yyCGH5NBDD83FF1/cEPQkc0Ogedt/AAAAAAAAAACofo7cWoSuu+66HHTQQdl///2z7bbbLvD7urq6rLjiinn22Wfz/e9/P7W18//tmXdc1qRJkzJkyJD07Nkz77zzTj766KP07ds3TZs2TTL3eK5523wAAAAAAAAAgCpScsoO/5rqYxGZNWtWfvazn+WCCy7Idddd1/B+qVRq+PNmm22W9dZbL7/5zW/yzDPPJPnrVp4kmTNnTgYNGpQbbrghSbL22mtnxx13zF577ZWmTZs2bPQR8wAAAAAAAAAANF7Kj0WgXC7no48+SrNmzfLRRx/l7bffzpAhQzJz5syG+GZeuDNs2LC8/fbbGT58eKZMmdKwlSdJHnjggUyePPkrt/uUy+UFNvoAAAAAAAAAAND4CHoWgUKhkNVWWy1dunTJ9ddfn3XWWScrrLDCfNcUi8WUSqVsueWWOe200/LEE09kp512ynHHHZef/exn2XHHHTN48OD07ds322+//Vd+BwAAAAAAAAAAjZ+VLgtJqVSab/vOG2+8keuvvz719fXp3bt3/ud//ifJ3M06fx/jnHjiifne976XCy64IGPGjMnvf//7rLvuurnnnnvSokWLf/g5AAAAAAAAAAAaP0HPN6xcLqdUKjUclTVr1qw0bdo06667bh588ME8++yzuf7663Pbbbdlr732mu/aeQHQUkstlW222SbbbLNNkmTatGkNIU99fX2KxaKYBwAAAAAAAABgMSXo+YYVCoUUi8W89dZbOf3001MsFrPlllumZ8+e6dy5c5o3b54xY8bk+uuvzy677JLmzZv/w2075XI5SdKiRYuUy+WUy+WG+AcAAAAAAAAAgMVTTaUHWBzdeOON6dSpU+rq6lIsFnPxxRdnwIAB+fzzz7P++uund+/e+fOf/5wrrrjin96nUCg0hD6FQqFhgw8AAAAAAAAA0EiVS16VfjUCCpFv2IwZM3Lrrbfm+OOPz6233pprr702t9xyS7788sscdthhSZKdd945W2yxRX79619nwoQJKRQKKZUaxz8w1WDWrFn5y1/+Mt9rdrm+0mMBAAAAAAAAAHwjBD3fsFKplBdeeCFrrLFGw3ubb755TjnllNx999154403stJKK2X33XfPrFmzcvbZZyeJ7Ttfw7nnnpsWLVrM97p5xpuVHgsAAAAAAAAA4BuhIvmayuXyP/39p59+mnXWWSeTJ09u2LpTW1ubb3/729lggw3ywgsvJEl22GGHDBgwIAMGDFjoMy9uTjrppEybNm2+1w+arV/psQAAAAAAAAAAvhG1lR6gsSkUCkmSu+++O1tvvXVWWWWV+X7fvn37dOjQIc8++2zGjRuXb3/720nmbu6ZMGFC2rVrlyQpFos5/vjjF+3wi4mmTZumadOm873XpFCs0DQAAAAAAAAAAN8sG3r+ha/ayDN69Oj0798/yyyzzHzv19fXJ0mOP/74vP/++znnnHPyzDPPZOrUqfnFL36RjTfeOOuss86/vD8AAAAAAAAAAEsuQc/f+Pu4plwuN2zk+Vsbbrhh1lprrYwePXq+94vFYsrlcjp16pQzzzwzn3/+efr27ZvvfOc7uf3223Puueemffv2833mq+4PAAAAAAAAAMCSy5Fbf6NQKGTOnDlZaqmlGn6eOXNmrrvuumyzzTbZbLPNkiQzZ85M27ZtM3PmzH94r1133TU77bRTXnvttXz66afp3r37ongEAAAAAAAAAAAaORt6/r/6+vqMGjUqXbt2bXjvvvvuy0033ZQ77rgju+22W1599dWUSqW0b98+s2fPztixYxs+O8/fbtxZeuml07lz54aYp66ubtE8DAAAAAAAAABQnUplr0q/GoElOuh59dVXs+GGG2bGjBkpFosplUqZNWtW+vfvn5YtW+ass85Knz59MmrUqGy00UY5/PDDc/HFFydJevfunddffz11dXUpFov/1vfV1lqIBAAAAAAAAADAP7dEFyblcjknnnhimjVrliRZf/31M3ny5IwbNy4//vGPc8455zRce+ONN+a2227L0KFD06JFi0yePDktW7ZMfX19isXifJt5AAAAAAAAAADgP7VEBz2bbrppNt1008yaNSvTpk1LsVjMwIED88ILL+SDDz5ouK5cLqdVq1Y54ogjUlNTkwcffDBPPfVUZs2alS+//DJNmzat4FMAAAAAAAAAALA4WeKO3CqVSvP9PH369Oy1117p2bNnVltttVx44YXp169fnnvuudx3330NnymX556hNmTIkJx88snp3Llz6uvrM3bs2EX+DAAAAAAAAAAALL6WqKCnvr4+NTXzP3Lz5s0zcODATJo0KXfeeWeSZMcdd8xmm22Wiy66KEkWOFKrc+fOufbaa7PMMss0BEJ/HwoBAAAAAAAAAMB/YokKeorFYiZOnJijjz46119/fd58880UCoV897vfTZ8+fXLyyScnSTbccMP06dMnn332WS688MIkyUsvvZTHH388SVJXV5d27dpl9dVXz/PPP58kC4RCAAAAAAAAAADwn1isK5S6urr5fn7sscfSpUuXvPDCCxk2bFh69+6dzz//PG3bts3++++fWbNm5ayzzkqSdOvWLT179swZZ5yRXr16pXPnznn33XeTJLW1tXnjjTfy+eefZ/PNN1/kzwUAAAAAAAAANFKlklelX43AYh301NbWJpm7Xeezzz7LpEmTcv755+fpp5/OyJEjUywWc+SRRyZJNt988wwePDiXXnpppk6dmjZt2uS4447LZZddls033zzvvfde9t9//yRzQ6Gjjjoqbdu2zfbbb1+pxwMAAAAAAAAAYDFUW+kBvmmlUqnh+KvXX389AwYMyJQpU9K2bdt8/vnn+eUvf5kkWW+99XLhhRfm+9//fvbff/9sv/326d+/fx588MEMGTIkd911V1ZaaaWGiCeZG/LU1NSktrY2d9xxR1ZYYYVKPCIAAAAAAAAAAIuxxWZDz7zjtWpqavLnP/85Dz/8cK644orsscceGTlyZLp165Z33303H3/8ccNndt111/To0SOnnHJK6urqsv7662efffbJuHHj5rsumRsK1dbWNsRCYh4AAAAAAAAAABaGRh/01NfXJ/nr8Vrjx4/PlVdemZ49e+bpp5/OIYcckq233joXX3xxNt1001x33XX55JNPGj4/bNiwjB07Nj//+c9TW1ub/fbbLxMmTEirVq3m+555IQ8AAAAAAAAAACxMjb5SKRaLSZJJkyZlhRVWyIknnpjDDjssW265ZebMmdMQ/CTJiBEjMnLkyDz66KMplUpJko022ij77bdfXnnllZTL5bRo0SI1NTUNG38AAAAAAAAAAGBRavRBz9SpUzNgwIDceeedOf7443PXXXdlxRVXzDHHHJPPPvssTz75ZMO1W221Vfr165dhw4bl3XffbXj/iiuuyFVXXZVCodDw3ryNPwAAAAAAAAAAsCg1qqDnb7ftzDNx4sS88sorOeGEE9KpU6css8wySZK99torG264YW6//fa8+eabDddfeeWVGTduXEaOHNmwpWdevPNV9wcAAAAAAAAAgEWp6oOeUqnUEN7MO15r6tSpDfHNRhttlGOOOSb19fVp1apVkmTmzJlJkjPPPDMvvvhiHn744cyePTtJ0qpVq9x///059NBDU1Mz/+PPuz8AAAAAAAAAwEJRKntV+tUIVGXQc/fdd6d9+/b54IMPUlNT0xDe3Hvvvdl4442z8847p1u3bnnmmWfSrFmz7Lbbbtlmm23y4x//OEmyzDLLpFQqZauttsquu+6aESNG5Pnnn2+4f69evRquAQAAAAAAAACAalJ1Qc+tt96ao446KocddljatWvXsIlnxIgROfzwwzN48OAceeSRad68efbcc8/cc889adeuXY499ti89NJLuffee5Mkc+bMSZKceuqpad26dVq3br3Ad/39hh4AAAAAAAAAAKi0qitaJk2alJYtW+akk05KqVTKxx9/nCT57W9/m5133jk/+tGPMmjQoNx3333ZZJNNctNNN2XixInp3r17dtttt5x66qlJkqZNm2bOnDlp165dRo8enQ022KCSjwUAAAAAAAAAAP+Wigc9N954Y6644oqGnzt06JCll146++23X1q1apWRI0fmiy++yMsvv5xtttkmSTJr1qwkydChQzN27NiMHz8+yy23XAYPHpy33347Z555ZpJkqaWWarjvvE0/AAAAAAAAAABQzSoe9DzzzDPZZJNNGn7+8ssv84c//CF33HFHhg8fnkMOOSTLLrtsOnbsmDvuuCNJUiwWkyQ77rhjmjRpkvHjxydJOnXqlMsvvzx9+/Zd4HvmfQYAAAAAAAAAAKpZbaW+uFQqpaamJiNGjEiSTJ8+Pcstt1yee+659O7dOxMnTsynn36aZO52nYMPPjgDBw7MqFGj0q1btyTJe++9lyZNmmSNNdZIkrRs2TKDBw9OkpTL5RQKhQo8GQAAAAAAAAAA/OcqsqGnXC6npuavX/3QQw+lR48eGTVqVIYPH55bbrklW265ZW699db86U9/SrFYzHbbbZdBgwZl4MCBOfvss/Pkk0/mhBNOSLFYbDiK62/vL+YBAAAAAAAAAKpOueRV6VcjsEiDnlKplFKptEBss9Zaa2XOnDkZOXJkPv744zRp0iR9+vRJkyZNctlllyVJWrVqlREjRmTvvffOAw88kMGDB+ezzz7Lww8/nPbt2893PzEPAAAAAAAAAACN1SIJesrlcsNWnpqamowbNy4jRozIhAkTMnPmzKy//voZMGBAnnvuudx///1Jku7du6dr164ZM2ZMnnjiibnD1tTkkksuyeOPP57HHnssDz30UFZdddXU19cviscAAAAAAAAAAICFbqEHPfX19SkUCikUCqmvr8/QoUOz7bbb5qKLLsrOO++cSy65JEkyZMiQrLTSSnnggQfyxz/+McViMbvuumvWWGONXHrppQ33K5fLadq0adZYY42Uy+XU19enWCwu7McAAAAAAAAAAIBFYqEFPSNGjEiShtjm5ptvzqmnnpq6urq89tpreeKJJ9K3b99ce+21GT16dJo1a5b99tsvf/rTn/LrX/86SdK5c+d06dIlEyZMyIsvvjh34Jq/jlwoFMQ8AAAAAAAAAAAsVhZK0PP6669nyJAhOf3005MkH3zwQR588MEMHz48c+bMyVprrZVVVlklBx54YDbZZJOG6wYOHJiNN944jz32WJ588skkyeDBg/Poo4+mc+fOC2NUAAAAAAAAAACoKv9V0FMqlb7y5w4dOuScc87JsGHD8tlnn6Vdu3b5wQ9+kHbt2uXLL79suH7dddfNwIED8+677+aaa65Jkhx88MF57bXX8sILL6RcLqd169Zp3br1At8FAAAAAAAAAACLo/8q6KmpqcmsWbPy8ssvN/ycJE2bNs2ee+6ZDh065JBDDkmSbLvtthk4cGDGjBmTV155peEe22+/fXbZZZdccMEF+eKLL7Lddtvl1ltvzY9+9KMUCoX5vgsAAAAAAAAAABZ3/1UlM3v27PTr1y8/+clP8v777ydJ6uvrkyTt27fPSSedlLvuuiu/+93vstxyy6VHjx5Zc801c9555zXco02bNtl5551TKpXy8MMPJ0m23nrrJAtuAAIAAAAAAAAAaNRKZa9KvxqB/yroadKkSfr27ZuPPvoo9913X5KkWCymXC6npqYmO+20U3r16pUhQ4YkmRvq9OnTJ7///e8brk+Srl275tlnn83uu+8+/3C28gAAAAAAAAAAsIT5r4uZAw88MO3bt88DDzzQcPTWvM06rVq1ypFHHpl33nknI0eOTJL07NkzHTt2zEknnZRyeW71tPTSS6dVq1Ypl8sN7wEAAAAAAAAAwJLoG1mBM2TIkEyePDn33ntvSqVSisViw9Fb3/rWt9KpU6e88cYbSZJ11103AwYMyNChQ1MoFOYLeAqFQgqFwjcxEgAAAAAAAAAANErfSNDTtWvXbLPN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XXbZJd/5zndy9913580331xkswMAAAAAAAAAVBtBD/+xTz75JCuuuGKeeOKJBd7v2rVr7rvvvhxwwAGpqanJGWeckeOOOy5Jsvfee2fFFVfMDTfckDlz5iRJWrZsmZ49e+aLL77Ik08+ucifBQAAAAAAAACgWtRWegAap3K5nJVWWim33HJLevXqNd/vXnrppXz44Ye58847s8kmm+TQQw/NnXfemb333js9e/ZMt27d0qNHj9x555257bbbsu+++yZJevXqlY4dO2bttdeuxCMBAAAAAAAAwMJXqvQANAY29PAfKZfLSeZGODNmzMgjjzySWbNmJUnGjx+fadOmZZNNNkmSlEql9O/fP9/97ndz0UUXJUkGDBiQVVddNVdccUWmTJmSJKmpqWmIeUol/wYDAAAAAAAAAJZMgh6+lvr6+iRz45t5LrroovTo0SOvv/56kmTNNddMkyZN8tRTTyVJ6urqkiQHHnhgXn311XzyySdZbbXVsttuu2WPPfbIiiuuuMD3/O39AQAAAAAAAACWJKoJ/m2lUinFYjFJ8uyzz+aBBx5Ikpx66qlZbbXVMmLEiJRKpay//vrZeOONc8011yRJmjRpkiR58cUXs+6662bZZZdNkuy7774ZOnRollpqqQo8DQAAAAAAAABAdRL08G+rqanJW2+9la5du2bAgAF55plnMmHChCTJhRdemOuuuy6jR49Ox44d06dPnzz55JM54ogj8txzz+Wxxx7Lgw8+mO233z7LLLPMfPedd3wXAAAAAAAAAABJbaUHoPF45ZVXss8++2SLLbbIDTfckJYtW6ZFixZJkgEDBuSiiy7KOeecky233DL7779/WrdunaOPPjqPP/54Jk+enEMPPTQ//elPF7hvoVBY1I8CAAAAAAAAAFC1BD0soL6+PjU1NQuENm+99VbmzJmT66+/PknyySefZPbs2fnkk0+yyiqrZMSIEdl8881z7733Zq+99kq/fv3StWvXfPjhh1l11VXTsmXLJHOP7qqpsRwKAAAAAAAAAOCrCHqYT7lcTrFYTJK89tprqauryyabbJKamppMmzYtSy21VE488cR89tln+fLLL/Pkk0+mefPmueqqq9KlS5cMGjQoJ510UrbYYotssMEGWWmllbLSSisl+WsoJOYBAAAAAAAAAPjHlBUkmRvyJHOPv/rss8/St2/fdOnSJbvuumt+8IMfZNq0adl9991z0EEH5Ve/+lXK5XI6deqUc889N6uvvnqOPvroJMlll12WJk2apK6uboHvKBaLjtcCAAAAAAAAYIlWLpW9KvxqDGzoIUkaQptHH300L774Ylq1apVnn302f/jDH3L44Yfnf/7nf3LyySfnmGOOyZFHHtmwxSdJxowZk9atW2fmzJlZbrnl8vbbb1fqMQAAAAAAAAAAGj1BzxKqVCqlUCjMtzFn1KhR2WmnnbL66qvn6quvzkYbbZSNNtoo77zzTm6++eZsscUW6d+/f4rFYiZPnpz6+vrceOONuffee3PRRRdlmWWWabhXfX39fNEPAAAAAAAAAAD/HkduLYHq6+tTU1OTQqGQv/zlLw3vd+vWLfvvv3/+8pe/ZKmllmp4/5hjjsnyyy+fkSNH5t13383EiRNz/vnnp2fPnrnppptyww03ZM8995zvO8Q8AAAAAAAAAAD/GUHPEqhYLObTTz/NAQcckD322COHH354br311iTJ8ccfn2bNmuWpp57K559/niSpra3NMccck5dffjkPPPBA1lxzzfTs2TPnnXde3njjjXTr1i3lcjnlcuM4Zw4AAAAAAAAAoJoJepZATz/9dDbZZJN8/PHH2XPPPdO8efPst99+GTVqVDp27Jh99tknd999d8aOHdvwme9///tp165d7rjjjnz44Yfp2bNndtlllyRJXV3dAsd3AQAAAAAAAADwnxH0LMZKpdJXvn/PPfdkr732yn333ZcDDzww3/3ud1NXV5df//rXSZKf/vSnmT17du69995MnTq14XMjRozIjTfemLZt2853v9ra2oX3EAAAAAAAAAAASxhBz2Lkj3/8Y5555pkkSX19fWpqvvpv79NPP50tttgikyZNyjbbbJODDz44F110Uf73f/83SbLsssvmRz/6Ue688848/PDDDZ9be+21s/rqq//DUAgAAAAAAAAA+BdKZa9KvxoBQc9i5PTTT0/fvn0zbdq0FIvFfPHFFzn55JNzySWXZMyYMUmSjz/+OK1atcoFF1yQDTbYIJtuumlefPHFHHPMMSkWi3nggQeSJAcffHA23njjrLjiigt8zz8KhQAAAAAAAAAA+O8pMxq5W265JX/84x+TJBdccEHq6ury85//POPHj8+GG26Y3/72t7nlllvSrVu3PPvss2nVqlU6deqUCRMm5IILLsiVV16ZVVddNUkybty4jBgxIi+88EKS5Le//W169+5dsWcDAAAAAAAAAFgSCXoasXK5nH333Tc/+9nPMmPGjKy66qoZOnRoLrzwwvzmN7/J0UcfnRdffDG//e1v06dPnxxyyCGZMWNGjj766Ky99tr51a9+lbvvvjuvv/56rrzyyuyxxx5p165d1llnnSRzN/E4XgsAAAAAAAAAYNES9DRSs2fPTqFQyM9//vP84he/yMsvv5wkGTp0aFZfffWcfPLJ6dChQ5JkxRVXzGWXXZa33norl112WVZeeeVceumlWXnllXPIIYdkn332yXnnnZezzjorI0aMSMuWLRu+x/FaAAAAAAAAAACLllqjEaqvr0+TJk2SJAcffHBWX331DB8+PFOnTk3Tpk1zxhlnJEmWWmqpJEmpVErbtm1z0kkn5cILL8yECRPy3e9+NzfddFNeffXVXH311Xn33Xez7777NlwPAAAAAAAAAEBlCHoaoWKxmCS59NJLc9ZZZ2WzzTbLgw8+mKeeeir19fXZZZddsuOOO2bYsGGZNWtWw5adU089Ncsvv3xOP/30fPnll0mSdu3a5dvf/naSpK6uLomtPAAAAAAAAAAAlaTcqFL/bEvO7Nmzs88++2TYsGFZeeWV07Jly7Ro0SIXXnhhJk2alCS54IILMmbMmNx5551JknK5nCQ588wzM3HixIZ452/V1tYuhCcBAAAAAAAAAODrEPRUqZqamsyYMeMrfzdlypS88MILueCCC3LooYfm8ssvz29+85uMHTs2d999d2bNmpXNNtssBx98cM4666x89NFHKRQKSZIf/OAHGTNmTJo3b74oHwcAAAAAAAAASJKSV8VfjYCgp0r9/Oc/z6qrrporr7wyU6dOTZLU19cnSaZOnZq33347Xbt2TTL3qKxOnTrliCOOyMUXX5w333wzydxtPH/6059y+eWXL3D/r9rQAwAAAAAAAABA5Ql6qtTo0aPTunXrvPXWWxk4cGBmz56dYrGYJGnTpk1WXXXV/PKXv0zy1+O59ttvv0yZMiV33HFH/vznP6dVq1b5zW9+k0MOOWSB+zteCwAAAAAAAACgOgl6qsy8OGfXXXfNWmutlR//+MeZMWNGBg8enAcffDBJ0rx58+yxxx65/vrr8/HHH6dJkyZJkt///vdZbrnlcs4552TChAlJkt69e2fVVVdNuVyuzAMBAAAAAAAAAPC1CHqqTE3N3L8ls2bNSvPmzdOqVatcddVV6dChQw466KA8+eSTadmyZQYNGpSmTZumb9+++fWvf53f/e53uf322zNy5Mjcc8892WKLLea7b6FQqMTjAAAAAAAAAADwNQl6KuD5559P8tdtPH9r3iadb3/723nkkUcybdq0bLrppjnjjDNSLBYzePDgnHrqqdl0001z2223pb6+PkOHDs2OO+6YlVZaKVtttVV23333Rfk4AAAAAAAAAAB8g2orPcCSpL6+PhdffHGGDRuWMWPGZL311kupVGrYypPM3aRTLpezxhprZMstt8wLL7yQmTNnZsiQIWnSpEkGDhyYK664In/5y19y4okn5plnnsmECRPSpEmTrLHGGhV8ukVn1qxZmTVr1nzv1ZfrUywUKzQRAAAAAAAAAMA3R9CzCBWLxWyxxRbZdNNNc8EFF+Tqq6+eL+aZp1AoZNasWZkzZ0769++fJDnqqKNy3HHHpXnz5tl2222z//77Z8aMGbnyyivToUOHFAqFho0/X3XPxcm5556bM844Y773Nlx+/WzccoMKTQQAAAAAAAAA/55yqVzpEWgEFu/yowptu+226dWrV5566qmMGjUqydzNPX+rXC6ndevW6dChQ1ZaaaVMnDgxp59+epo3b54k6dGjRx555JFcc801WWqppVIoFJLMDXkW95gnSU466aRMmzZtvlfHFutWeiwAAAAAAAAAgG/E4l9/VNi8rTnz/lwsFtO7d+9stNFGufDCC5PM3dxTLi9Y4G266aZZddVV5/vdvD9vvPHGSRaMgZYETZs2zfLLLz/fy3FbAAAAAAAAAMDiQtCzkJTL5dTX1zdszJkxY0bDnzt27Jhdd901kyZNytVXX51k/vBn3sadpk2b5sMPP8ysWbMW+N08xaKQBQAAAAAAAABgcSLoWUgKhUKKxWImTpyYfffdN4MGDcrZZ5+dl19+OUmyww47ZNttt83Pf/7z/PnPf06xWGyIeub957e+9a188cUXS+QWHgAAAAAAAACAJZWgZyG6+OKL861vfSt1dXXp2LFjnnrqqfzwhz/M559/ntVXXz0777xzampqMmzYsCR/3b4zb5PPFltskffeey+rrLJKxZ4BAAAAAAAAAIBFS9DzDfjb47LmefvttzN69OjceOONufXWW3P22Wdnq622yrPPPpvTTjstSbL99tunR48e+c1vfpPx48enUCjMt42ntrY2SVJXV7doHgQAAAAAAAAAgIoT9PyX6uvrGzbqzJkzp+H9tddeOwcffHD69OmTsWPHpkuXLrnuuuuy11575eqrr86rr76a5ZdfPr17906LFi1y4oknJkmKxeIC3zEv7AEAAAAAAAAAYPGnFPkPlEqllMvlFIvFFIvFTJkyJSeffHKaNGmSHXbYIf3790+S7LLLLnn33XdzxBFHZMstt8yoUaPy3nvv5dFHH82ZZ56ZO++8M1tttVX222+/rLzyyhV+KgAAAAAAAABgoVvwECBYgA09/6bhw4fntttuS5LU1NQ0bNL53e9+l169euXdd9/NlClTGjbwfPnll0mSO++8M3/+859zzjnnpFmzZpk5c2aWWmqp3H333bnttttSKBRy8MEHZ7fddqvYswEAAAAAAAAAUD1s6Pk3TJ8+Pa+88kqGDh2aZO6GnokTJ+awww5L27ZtM2DAgPz4xz9Okhx55JG59tprs8Yaa6RHjx5p0aJFZs6cmQ8//DDLLrtsfvWrX2XvvffOJptsku22266SjwUAAAAAAAAAQBUS9PwLpVIpyy23XK699tokySeffJKVVlopa6+9dqZPn55Ro0bluuuua7j+Jz/5SXr37p2RI0emS5cu6dKlS9ZYY41873vfS6FQyHLLLZdf/vKX2WyzzZIk5XI5hUKhEo8GAAAAAAAAAEAVcuTWP1EqlVJT89e/RNdee20GDBiQp556KklywQUXZMUVV8yHH36YOXPmJEnatGmTAw44IE8++WQefvjhbLzxxrn33ntz+umn55xzzsnrr78u5gEAAAAAAAAA4B8S9HyF+vr6JGmIed57770kyWqrrZZPP/00jzzySObMmZNtt902PXr0yAMPPJDf//73DZ8/4ogjssoqq+Sqq67K66+/npVXXjkHHXRQ9t133yRJXV1dkoh5AAAAAAAAAABYgKDn75RKpRSLxSTJ+++/nx/96Ec54ogjUl9fnx49euR73/teHn/88Tz00ENJkrPOOivvvvtufv3rX2f69OkN9/nhD3+YNdZYI6usssoC31Fb66QzAAAAAAAAAAC+mrLk79TU1OTNN9/MMccck1VWWSVjxozJ7Nmzc++992aPPfbID3/4wwwaNCj3339/vvOd72TttdfOoEGDMnLkyHTp0iU9e/ZMkuy+++7ZfffdK/swAAAAAAAAAEBVKZfKlR6BRmCJ39Az73iteV5//fX07t07rVu3zn777ZfDDz88X3zxRW644YZMmTIlHTp0SN++ffPiiy/m/vvvT5KcdtppmTx5ckaOHJkZM2b80/sDAAAAAAAAAMA/s8QGPfNCm3nHa9199915//33M2HChMyYMSPnn39+tt9++xx77LE544wzMmXKlPziF79IkgwZMiRt2rTJQw89lFdffTVLLbVUfvWrX+W8885Ls2bN5vueefcHAAAAAAAAAIB/xxIb9MwLbaZPn56ePXvmxz/+cT799NOMHz8+a665ZpZaaqmGa/fZZ5+svPLK+dWvfpXXXnstyy67bPbee+8899xzefXVV5MkXbp0SYsWLVIqlSryPAAAAAAAAAAALB6W2KCnrq4u++yzT84999yss846efrpp9OpU6d06dIl48aNy4QJE1Iul1Mul7P88stn2223zYsvvtiwpWfvvffO7bffnr333nu++9bULLF/SQEAAAAAAAAA+AYsEfXJvOO1/lZtbW2aNWuW8847L3PmzMnKK6+cJPne976XLbfcMqeffno+/vjjFAqFJMmUKVOyxRZb5OWXX85TTz2VJNlqq62SJOVyeRE9CQAAAAAAAAAAi7vFOugplUoplUoNx2t98MEH8/3+7LPPTrt27VIul1NfX98Q5lxzzTV5/vnnM2jQoIwYMSKnnXZaXnnllQwZMiTvvfdeJk+ePN995kU/AAAAAAAAAADw31psgp5SqTTfz+VyOTU1Nampqcmzzz6b7373u9lll13So0eP/Pa3v02SrLzyyjnhhBNy6623ZsKECSkUCimVStlggw1yxx13ZJVVVsnll1+ee+65Jz/5yU+yzz77ZNKkSZk6dWolHhEAAAAAAAAAgCXAYhP01NTUZNq0aXnppZca3quvr895552X3XbbLV26dMmpp56aTTfdNHvttVdeeOGFJMkRRxyR9u3b58wzz2yIgJKke/fuue666/LEE09k/Pjx2WGHHXLzzTenQ4cO6d69e0WeEQAAAAAAAABo5EpeFX81AotN0PPggw+mTZs2Oe644zJ58uQUCoVMmTIlM2bMyLXXXpvzzz8/ffv2Tbt27TJt2rRccsklmTp1ampqajJs2LDcfffdefzxxxvuN+/4rXK5nJtvvjmDBg3K4Ycfnn79+mW99dar1GMCAAAAAAAAALCYa3RBz7zQ5u+NHTs2s2fPTk1NTW6++eYkSbt27bLrrrumT58+efTRR7PBBhvklltuyfnnn59f/vKXGTVqVOrq6tK7d+906dIlRxxxRGbNmpUkKRQKSZImTZrk1VdfzfTp0/Piiy/mpz/9acPvAAAAAAAAAADgm9aogp5yudwQ05RKc3cgzZ49O0nSq1evrLzyymnRokWefvrphqO3tthii0ydOjWnnXZa+vfvn8ceeyzHH398Ntpoo9xwww156623kiTXXnttLrvssjRt2nS+71x++eXz05/+NPfee6/NPAAAAAAAAAAALHSNKugpFAqZOXNmTjnllFx44YVJ5m7QSZK6urp079493/nOd/LRRx/l9ttvb/jcr3/960yePDn77LNPll9++bzxxhspFAp55JFHct9996W+vj5rr712dtxxx6/83mWXXXbhPxwAAAAAAAAAACSprfQAX0ddXV2OO+64XHnllQ0/77333llzzTWzyiqr5KmnnspZZ52Vv/zlL3n88cfz0EMPpWfPnmnfvn3ef//9jB8/Pk2bNs0VV1yRo446Kquvvnq22267FIvFCj8ZAAAAAAAAAADM1aiCntra2vTp0yfvvPNOPv/884wZMyYvv/xyzjjjjHTs2DHrrbde7r///hx66KEZM2ZM7rrrrmyzzTbp3r17dtlllwwdOjSfffZZ1llnnRx11FFZd911k8w9vqumplEtKwIAAAAAAAAAYDHVqIKeJOnZs2dGjRqVP/zhD+nRo0cmTZqUfv365dJLL80GG2yQmTNnZrXVVkuvXr3yq1/9KnfffXf233//3HrrrZkwYUKmTZuWbbbZZr57inkAAAAAAAAAgEWhXKr0BDQGjbJk2WeffVJfX59x48bl/PPPT69evXLVVVflyiuvzCuvvJIkGTx4cFq3bp0bbrgh7777bpZeeulsvPHGDTFPfX19JR8BAAAAAAAAAAC+UqMMer71rW+le/fuefXVV3PPPffkf//3f7P//vunUChk6aWXzuzZs9OqVav069cvO+64Y9q0abPAPYrFYgUmBwAAAAAAAACAf67RHbk1z1577ZXnnnsu1113XbbaaqvsvPPOefPNN7PWWms1XDNo0KAKTggAAAAAAAAAAF9fo9zQkyTt2rXL7rvvnk8//TQ33nhjkmSttdZK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AAAAAAAAANVB0MNcUSqVUi6XG2KecePGZamllsoXX3yRyy67LIMHD87aa6+d4cOH5+ijj84yyyyTDh06VHhqAAAAAAAAAJi7yqVCpUegEXC+EXNUqVRKfX19ampqUltbmzFjxqRv377ZZZddMm7cuEyZMiUrrLBCpkyZkl69emWXXXbJzjvvnFdffTXbbLNNpk2bliQpl8sVXgkAAAAAAAAAQGUIevivnXXWWbnrrruSJDU1Namrq0u5XM5BBx2Un//85xk5cmTee++9LLzwwll00UUzZsyY/OpXv8oCCyyQV155JSeddFIWXHDB/POf/8zAgQPz5ZdfplBQJAIAAAAAAAAATZOgh//KRx99lNGjR2e11VZruPfAAw+ktrY2L7/8cp555pmce+65WXHFFVNbW5tWrVrl8MMPT8uWLbP77rs3HLM1adKkXHjhhRkzZkwmTpxYqeUAAAAAAAAAAFRcXaUHoHEplUqpqfmmAysWi1lyySVz+eWXJ0kmTJiQdu3apU2bNhk2bFh23nnnJMlFF12URRZZJC1atEiSHHTQQbn99tvz+9//Pn/+85/TpUuXXHXVVfnJT36Syy+/PEsvvXRlFgcAAAAAAAAAUAXs0MMPUiqVknxzpNaECRNy2WWX5fHHH294/fzzz8+vfvWrvP7661l//fUbYp4keeWVV7LBBhskSaZOnZokueaaa3L88cenY8eO+cc//pHjjz8+zz33XNZcc815tygAAAAAAAAAgCpkhx5+kFm78hx77LG54oorss4662T99ddP586ds9hii2XppZfOp59+mrvvvjurrrpqkqS+vj5TpkzJxIkT07FjxyRJq1at8tVXX6VVq1bZdddds+uuu872PcViMbW1tfN2cQAAAAAAAAAAVcQOPfwg48aNy9Zbb50HH3ww1113XS6//PIcccQRWWyxxZIkO+20U9ZZZ508+OCDefLJJ5MkdXV1+eijj/L+++9n4403TpIMGjQoa665Zh588MHZPn/WDkBiHgAAAAAAAACgqbNDDz/Is88+mw8//DB33313OnXq9L3PHHzwwdl///1z2223Ze21186CCy6Y119/PWussUYeeuihnHTSSZkyZUouuuii9O7de7b3ztoBCAAAAAAAAADmZ+VSodIj0AioKPhflcvlPPPMM/nJT36ShRZaKEny9ddf57HHHsuDDz6Ye+65J1999VU6d+6cbbfdNiNHjsy9996bJPnyyy/z3HPPZe+9986ee+6ZcePGNcQ85XK5UksCAAAAAAAAAKhaduihwbvvvptlllkm5XI5hcI3ReCsn5dddtnccMMNOf7447Pooovm/vvvT+vWrfPKK68kSdZbb73cf//9OeSQQzJixIjcdddd2WSTTbLqqqvmnHPOyYEHHpgFFlggSVJfX5+6urqG7wAAAAAAAAAA4FuCniauXC5nxowZ2WeffTJjxoz8+c9/zk9+8pPZop4k2X///fPFF1/kvvvuy8SJE7PLLrvk5z//eVZbbbV8+umn6datW2644Ybssssu6d27d84666y8+OKL2WqrrbLOOusk+Tbkqavz1w4AAAAAAAAA4D9RVjRh48ePT7NmzbLYYotljTXWyK233pp77rkne++9d0PMUygUUiqVUlNTk6OOOioHH3xww047s7Rr1y6dOnXK3//+9+yyyy45+OCD84tf/CJdunRpeKZcLgt5AAAAAAAAAAB+gJpKD0BlvPTSS9l+++1z1llnJUkOP/zwLL744rn77rvz1ltvJUlKpVKSpKbm278mLVq0+M5njRw5Mq1bt07//v0b7s2KecrlcpI4XgsAAAAAAAAA4AcS9DRRP//5z7PWWmvl+eefz3PPPZfmzZtnv/32y7/+9a/cfvvtSWYPeWapra1Nkvzzn//Myy+/nCOPPDK77757evbsmdVWW+07zwt5AAAAAAAAAAB+HEFPE1QqldKqVav86le/SrFYzJVXXpkk6d27dzp37pyHHnooI0aMSPLtDjv/7oYbbshBBx2UvfbaKyNHjszdd9+dgQMHplmzZvNyGQAAAAAAAADQ6JTLrkpfjUFdpQdg3ikWiw077CTJpptumocffjgPPfRQ7rjjjuywww75zW9+k4MOOii33npr1lprrSywwAIpl8uz7bSz4447ZqGFFsrCCy+cjTbaKMn3H88FAAAAAAAAAMCPp75oAmbFNrNinsmTJ6dYLCZJ+vbtm0UXXTTXX399pkyZki5dumTLLbfMyJEjc//99yeZ/discrmc5s2bp2fPng0xT7FYTE1NjZgHAAAAAAAAAGAOUGDMR77veKzk211zLr/88qyxxhrp06dPevbsmTFjxmT11VfPDjvskDFjxuSqq65KkhxyyCFp0aJF7r777owbNy7Jt1HQv8c9s/z7rj8AAAAAAAAAAPx3BD3ziVKp1BDbzJw5s+F+fX19yuVyjjzyyJx66qk57LDDctxxx6VVq1bZfvvt8/LLL6dfv35Zfvnlc+edd2bMmDFp3759+vfvn1GjRuWGG25I4igtAAAAAAAAAIB5RaXRyM3alaempiYTJkzI0Ucfnd/97nc58sgjkyR1dXX57LPP8swzz+TSSy/Nvvvum27duqV58+b54IMP8sknn6Rt27bZeeed89VXXzXs0rPffvtlnXXWSefOnSu2NgAAAAAAAACApkjQ08jN2pXntNNOy9JLL533338/K664YhZbbLF8/fXXSZJ33nkn7733XrbaaqsMHDgwP/nJT/L1119n5MiR2WKLLZIkO+ywQ9Zee+3cfPPNeeqpp5IkV199dXr06FGZhQEAAAAAAAAANFF1lR6A/96FF16YG264Iddff3122GGH77zesmXLLLTQQmnTpk2WW265XHfdddl+++2TJKNHj85HH32UTTbZJDvuuGPatm2blVdeueG9pVLJcVsAAAAAAAAAAPOQoKeR+/jjj3PhhRemT58+2XbbbWd7bVaMs8QSS2S11VZLTU1N7r333iy55JINz1x99dWZMGFCNtxww2y66abZdNNNZ/sMMQ8AAAAAAAAAzDnlUqHSI9AIqDUaubFjx+a9995L//7906xZs5TL5YbXZsU47du3zy9/+cs0b948++yzT2655ZY88sgj2WqrrfK3v/0t22yzTerqvm27SqXSPF8HAAAAAAAAAADfsENPI/evf/0rpVIpEydO/N7jsWbd69OnT5ZeeukcfPDBOf300/Pll19m3XXXzc0335zWrVvP9h678gAAAAAAAAAAVI6gp4pNnDgxG2+8cQYNGpSePXt+7zPrr79+WrZsmSeffDIbb7zxd16vqanJ/fffn9ra2myxxRZ58sknM3PmzEyZMqXh6K1isZja2tq5uhYAAAAAAAAAAH4YW7FUsUUWWSRLLLFETjvttEyZMuV7n1l44YWz00475YILLsg//vGPFIvFJGk4emvcuHG58sor8+GHHyZJWrVqlTZt2mTJJZdMqVRKqVQS8wAAAAAAAAAAVBFBT5W7+OKL8+KLL+baa6/93tdbt26dAw44IM2bN8/BBx+cUaNGJUmmTp2a0aNH58gjj8ykSZOy+eabf+e9NTU1jtcCAAAAAAAAAKgyao4qM2tnnVmWX375HH744TnllFMybty4733P+uuvnyuvvDKvv/56Nt9883Tu3Dn9+vVL165dM2PGjFx33XXp2LHjvBgfAAAAAAAAAID/Ul2lB+Ab5XL5Px5/deyxx+aaa67Jueeem3PPPfd739ujR4/8/e9/z8MPP5xPPvkkkyZNykknnZSuXbsmSUqlkt14AAAAAAAAAKDCyuVCpUegERD0VEi5XE6hUEipVEqhUEihUEhtbW3GjRuXa665JiuvvHJWWWWVLL/88mnTpk1OOeWUDBgwIHvssUc6d+4822cVCt/8y96xY8fsueees71WKpWSRMwDAAAAAAAAANBIqDwq4Nxzz80FF1yQ5JvQZlaQc+KJJ2bllVfO7bffnuOOOy4bb7xx3n777STJPvvskzXXXDN/+MMfUl9f/4O+Z9auPGIeAAAAAAAAAIDGQ+kxj33xxRf54osv0qNHj4Z7M2bMyGWXXZbHHnsst956a0aNGpXXXnstzZs3zx//+Md88MEHSZKzzz47d999d+6///4f9F1CHgAAAAAAAACAxkfxMQ+MHz8+e+yxR1566aUsvPDCOfXUU7Paaqtl3LhxSZLmzZunffv2OfPMM9OjR48888wz6datW7766qvccMMNue+++zJz5sx069Ytu+22W37729/miy++qOyiAAAAAAAAAACYKwQ988A777yT1157LYMHD264N2zYsPTs2TMPPPBAkmTTTTfNBhtskKuuuiq/+tWvsuaaa2bChAnp379/Lrjggvzzn/9Mkpx88smZNm1aw649AAAAAAAAAADMXwQ988AGG2yQvffeO08//XTuvffeJMlSSy2VxRZbLMOGDUupVEqbNm3y9ddf56abbsovf/nLhvhn4YUXzuuvv56hQ4dmypQp+dnPfpaxY8dmtdVWq+CKAAAAAAAAAACYWwQ9c0m5XE6SlEqlFAqFbLXVVlljjTVy3nnnJUk22mijbLHFFnn99dfzt7/9LUkyYcKEPPzww1lnnXVSW1ubadOmJUl22223vPvuu6mrq0uS1NTUpFgsVmBVAAAAAAAAAADMbYKeuaC+vj6FQiHJN/FNkqy44orp1atXxo8fnwsvvDBJ0rdv33Ts2DHDhg3L+PHjs/TSS6dHjx456KCDcsABB2TttdfOO++8k6FDh+amm25KixYtGr6jtrZ23i8MAAAAAAAAAPivlEuuSl+NgaBnLqirq8vXX3+dgQMHZujQoXnwwQeTJFtvvXW6deuWv/71r/nkk0+yzDLLpGfPnvn0009z5ZVXJkluuOGG7Lvvvvnwww/Tv3//3H333WnWrFmS2JUHAAAAAAAAAKAJEPTMQbOO2br22mvToUOH3Hfffbn99tvTt2/fDB48OK1bt86uu+6aurq6nHHGGUmSnXfeOZ07d87w4cPz3HPPZaGFFsppp52WO+64I8cdd1ySb0Meu/IAAAAAAAAAAMz/BD1zUKFQyOTJk3PppZdm4MCBGTFiRO6888707Nkzp512Wp566qmst9562W677XLffffl+eefz0ILLdSwS8+zzz6b5Jtjumpra1MqlVIul4U8AAAAAAAAAABNSF2lB2hMJk6cmNra2rRp0ybTp09PixYtvvPMPffckylTpuTggw/Om2++maOOOiojRozIcccdlw022CAtWrRIz5498+yzz+bUU0/N7bffnp49e+ZnP/tZVl111dk+q6ZGb/V9pk+fnunTp892r1guprYgfAIAAAAAAAAAGj/FyA/09NNPp0ePHrn66quTJC1atEixWMy9996bf/zjHw3PtW7dOu+//36OOuqorLfeellkkUXy0ksv5aijjkqLFi3y+eefp0uXLunevXvefvvtvPXWW0nSEPOUSqV5v7hGZtCgQWnbtu1s1/OT3qz0WAAAAAAAAADAfOiiiy7Kz372s7Rs2TJdunTJk08++R+f3WuvvVIoFL5z/c9NXv43gp4faO2110779u3z+OOP55133slDDz2URRZZJIcffnh+8Ytf5NJLL83UqVOz4oorZokllsi1116bZ555Jtdcc006deqU5Jt/wLfcckuSZM8998zTTz+dlVZaabbvsSvP/+7YY4/NpEmTZru6tF250mMBAAAAAAAAAPOZG2+8Mb/97W9z/PHH58UXX0y3bt2yzTbbZOzYsd/7/J/+9Kd89NFHDdf777+fRRddNL/85S9/1PeqR75HuVye7fdp06alefPmOeigg/Kvf/0rw4YNy7Bhw3LBBRdk1KhR6devXy6//PLcfPPNWWGFFbLNNtukWCymvr4+U6ZMSalUytVXX51LLrkkX331VYrFYhZffPG0bt06xWKxQqtsvFq0aJE2bdrMdjluCwAAAAAAAIDGoFQuuCp8/RjnnXde9t133+y3335ZeeWVM3jw4Pz0pz/NxRdf/L3Pt23bNksssUTD9dxzz2XixInZe++9f9T3Cnq+R6FQyOeff54hQ4YkSVq2bJkvv/wyvXr1SteuXXPJJZfk3XffzW677ZZFF100gwcPztJLL51hw4Zl3LhxOfroo7Pppptmo402yuabb54NNtgghx12WA4//PAcfvjhqa39Nj75958BAAAAAAAAAJi7pk+fnsmTJ892TZ8+/TvPzZgxI88//3y23HLL2e5vueWWGTly5A/6riuuuCKbb755w+lOP5Sg599cdtllqa+vT5I8/vjjOfTQQ/PMM8/kd7/7XdZaa6289dZbOeyww7LgggtmxowZadasWZJvgp/ddtstn3zySf76179mscUWy7Bhw3Lddddl3333zR577JHPPvss++yzT5KkVCpVbI0AAAAAAAAAAE3ZoEGD0rZt29muQYMGfee5CRMmpFgspn379rPdb9++fcaPH/+/fs9HH32U++67L/vtt9+PnrHuR79jPvXMM8/kwgsvzI477pjFFlssffr0ySqrrJIePXpkhRVWyFVXXZWVVlopSdKvX7/ccccdufXWW7PjjjsmSfr06ZPhw4fn8ccfz/rrr58ePXqkZ8+es31HfX196urqUlOjowIAAAAAAAAAqIRjjz02RxxxxGz3WrRo8R+fLxRmP6arXC5/5973ufLKK7Pwwgund+/eP3pGZcn/b7311ssrr7ySxRZbLEkyevTofPrpp5k6dWqOOOKIbLjhhg3bKx100EFp3bp17rjjjowbN67hM/bYY498+OGHefbZZ7/z+eVyOXV1+ikAAAAAAAAAgEpq0aJF2rRpM9v1fUFPu3btUltb+53deD755JPv7NrzP5XL5QwdOjS77757mjdv/qNnFPT8D4ceemhOP/30rLjiinn33XdzzDHH5Mgjj8y0adPSokWL1NfXZ/HFF0///v3z+uuv5+6772547y9+8Ytcc801OfbYY7/zuT+kzAIAAAAAAAAAoDo0b948Xbp0yfDhw2e7P3z48GywwQb/z/c+/vjj+ec//5l99933//TdTTboKRaL33t/gQUWyFlnnZX33nsvrVq1yj777JNmzZrlxBNPnO19++67bzp16pRrrrkmr7zySsP7u3TpkiQplUpzeQUAAAAAAAAAAMxNRxxxRC6//PIMHTo0b775Zg4//PCMHTs2Bx54YJJvju/aY489vvO+K664Iuutt15WW221/9P3Nsmgp1wup7a2Nkny2muvZezYsQ2vnXzyyWnXrl1OP/30JMlyyy2XY489NhdccEHGjh3bsMXSzJkz89vf/jarr756Onbs+J3vqKlpkn+0AAAAAAAAAADzjX79+mXw4ME55ZRTsuaaa+aJJ57Ivffem06dOiVJPvroo9m6kySZNGlSbrnllv/z7jxJUvdfTV3lisViQ7iTfBPhNGvWLIVCIe+991769++fN954I61atUqfPn1ywAEHZLXVVstpp52W3XbbLXvttVfWX3/97Lrrrvnb3/6WXr165dBDD82FF16YTTbZJIMHD063bt0quEIAAAAAAAAAoDEplwuVHoEfacCAARkwYMD3vnbllVd+517btm3z9ddf/1ffOV9uIzPruKva2tpMmzYtd911V5KkWbNmmT59ep566qlceOGFWWWVVfLYY4/lpJNOylNPPZXf//73+eSTT9KvX79stNFGOeGEEzJz5swsuuiiueaaa7LUUkvlwgsvzDbbbJPBgwd/5/sAAAAAAAAAAOC/NV8FPeVyOeVyueG4qzPPPDMLL7xw7rjjjsyYMSNJ8pe//CUbb7xx7rvvvhx66KFZffXVs//+++eEE07Il19+mSFDhiRJzj333IwYMSK33HJLkmTZZZfN9ddfn1GjRjUcxzUr5HG8FgAAAAAAAAAAc8p8VaIUCoUUCoXce++9WXrppfOXv/wl119/fS6//PI0b948SXLYYYdlrbXWyqeffpqWLVs2vHfLLbfMaqutlmeffTaffvppunTpkn79+uWggw7K5MmTk3yzJVLLli1TLBZnC4cAAAAAAAAAAGBOma+KlKlTp2bAgAHp2bNnjjzyyLz99tvp06dPw+uzzicbOHBgJkyYkBdffDH19fVJktatW2f55ZfPO++8k0UWWSTJN7v0XHXVVWnTps1s31NbW5tCwZl2AAAAAAAAAADMefNV0FMsFjN+/Ph069Ytffr0SV1dXZJvjuI65phjcvHFF6e+vj7bbLNNunfvnrPOOitvvvlmw/snT56c9u3bZ9q0aUmSdu3apVevXhVZCwAAAAAAAAAATdN8E/SUSqUstNBC+fWvf50kufzyyxv+s0OHDhk+fHh69OjRcEzWpZdemueeey79+vXLOeeckxNOOCGnn356dt555yy00EIVWwcAAAAAAAAAAE1bXaUHmFNmhTpbb711Hn744dxxxx256qqrUldXl1NPPTV77bVXamtrk3yzk89yyy2X4447LqeffnomTpyYd999Nw888EC6d+9eyWUAAAAAAAAAAPOxcqlQ6RFoBOabHXqSb3bpSZL+/ftn4YUXzoILLpiRI0dm3333TaFQSLlcTpKGsOfkk0/OggsumGbNmuWaa65J9+7dUywWGz4HAAAAAAAAAADmtUYV9BSLxe/cmxXpJN/u0tO5c+dsv/32ad26dR588MGG1wuFQj7//PMcfPDBGT58eGpra3PmmWdm8ODBeeaZZ1IqlVJbW9vwOQAAAAAAAAAAMK81mnKlXC437Kzz3HPP5dVXX820adNSKMy+FdWs3XV22WWX/PSnP82NN96YKVOmpKamJmeffXY6deqU559/PiuttFKSZMCAAWnVqlVOPfXUTJ06dd4uCgAAAAAAAAAA/oe6Sg/wfcaMGZN27dqlTZs2KZVKqampSaFQyKuvvpp99tknkyZNysyZM9OlS5fstdde6dmzZ4rFYsPuOuVyOR06dEjv3r1zySWXZL/99stLL72UyZMn59prr80OO+yQJJkxY0aaN2+eO++8M19++WUWXHDBCq8cAAAAAAAAAICmrup26HnzzTez44475ve//32SNOzA89FHH+XQQw9Nly5d8vTTT2f48OFp165ddt9990yePLlh955/t+OOO2bJJZfMQw89lP79++fDDz9siHmKxWKaN2+eJFlnnXWy2WabzaMVAgAAAAAAAADAf1Z1O/QsvfTS2WGHHXLbbbflpZdeypprrpkkefbZZ/P+++/n0UcfTZJcffXVueGGG7LxxhtnypQpadOmTcNnFAqFlEqltGrVKqeffno6dOiQli1bJknq6+tTV1f3vQEQAAAAAAAAAABUWtXs0FMqlVIqlbLgggumV69eWWqppTJo0KCG199666306NEjd955Z1ZYYYVceumlGTp0aO68884sueSSmTFjRpKkXC4nSWpqvlnacsstl5YtW6ZYLKZcLqeuruoaJgAAAAAAAAAAaFAVQU+5XE5NTU1DhNOlS5f07NkzL7/8cm6++eYkSadOnXLZZZdl3333zT777JMXXnghO+64Y0qlUu65557cdNNNSb49out/qq2t/Y+vAQAAAAAAAADMC+Wyq9JXY1DRoKe+vj7JtxHOaaedlquvvjpJssUWW6RLly654IILUiwWs8suu2SttdbKOuusk0MPPbThCK0XXnghf/nLX/Lll19m5syZlVkIAAAAAAAAAADMIRUNemYdf3X11Vfn3HPPzbXXXpvTTz89SfLzn/8822+/fSZNmpTzzjsvSXLuuefm6aefzlZbbZU//OEPGTBgQLp3754ll1wyu+++e5o1a1axtQAAAAAAAAAAwJxQ0aBn3Lhx6dq1a0455ZRMnjw5bdu2zejRo3PqqacmSbp3757u3bvn2muvzdixY7PpppvmlltuyTrrrJN33nknH3zwQR588MFceumlWXDBBVNuLPsiAQAAAAAAAADAf1A3r76oXC43HK01y4MPPpgZM2bkscceS8eOHTNgwICccsopOf/887PvvvtmqaWWynbbbZfnnnsu55xzTi644IKGyGf69Olp0aJFkqRUKiVJamoq2icBAAAAAAAAAMB/ba4XMMViMUm+E/MkyXvvvZcZM2akY8eOSZL27dtnv/32S8eOHXP00UcnSbp165Ztt9021113XUaNGtXw3lkxT7FYTE1NjZgHAAAAAAAAAID5wlyrYGbtmlNbW5skufrqq3PMMcfkpptu+vbLa2qyxBJL5LXXXmu4t+aaa2aDDTbI3/72tzz77LNp2bJlevTokWOOOSYrrbTSd75n1ucDAAAAAAAAAMD8YI4dufXkk09mwoQJ2WGHHWbbMWfq1KnZe++9M2LEiKy44ooZPHhwnn766Zx77rnZfPPNc+211+aRRx7JqquumkKhkEKhkFKplNra2hx55JF54oknst5662W99dabU6MCAAAAAAAAAFREufTdE47gf5pjQc9f//rXrLjiig0hT7lczkEHHZS2bdvmJz/5SV577bW0bNkyd911V/r27ZsePXpk2223bYh6ZsyYkf333z9vvPFGPv3005x//vm58MIL89BDD2XzzTdPuVxO8v1HdwEAAAAAAAAAwPzi/3Tk1qy45t9/Hjp0aI455pgkSbFYTKFQSJs2bXL22Wfnq6++Stu2bdOiRYvsvPPO6dWrV/7whz9kxowZOfnkk9OzZ8+ccMIJ2WijjbLhhhtm3XXXzcYbb5wpU6Zk4sSJSdKwew8AAAAAAAAAAMzPfnTQUyqVGsKaESNG5PHHH8+UKVOSJP/85z+z5ppr5umnn06SnHjiiVlhhRVSX1+fUqnUEP8MHjw4r7zySi677LK0a9cuf/jDH/LMM89k0KBBGT16dI499thMnTo1NTU1WW655ebUWgEAAAAAAAAAoOr96KCnpqYmL7/8cjbeeOPstddeueWWWzJ69OgkyfLLL58PPvggl112Wb744ou0bt06J5xwQm688ca89NJLKRQKKZfLWWaZZXLkkUfmxBNPzD/+8Y8kSefOndOjR48ssMACueuuu9K/f/907949K6644pxdMQAAAAAAAAAAVLEfHPTM2l1nyJAh2WKLLbLGGmvkrrvuyqGHHpq11lorpVIpSXLdddflmmuuyVNPPZVyuZzdd989Xbt2zYknnpiZM2c27O7zxz/+MQsvvHA+/fTThu/4+OOPs8cee+TAAw/MXnvtlauuuioLLbTQnFwvAAAAAAAAAABUtbof+mChUMiXX36Zm2++OSeccEIOPfTQ2V6vqfmmDdpqq62y6aab5swzz8xaa62VDh065Oyzz0737t1z9913p0+fPimXy2nevHnefvvt1NbWNnzGMsssk5NOOilrr712FlxwwTm0RAAAAAAAAAAAaDx+1JFbTz/9dF577bVssskmDffGjBmTN998M88991w+/PDDJMlf/vKXjBgxInfeeWdmzpyZDTbYIL169cpBBx2UKVOmNOzSU1tbm2KxmOTbHYC6desm5gEAAAAAAAAAoMn6wTv0JMlGG22UadOm5fzzz8+2226bO++8Mx9++GE++eSTvPXWW1l33XVz0UUXZa211sqBBx6Y8847L5tssklWWWWVDBkyJM8999x3Yp1ZO/TMinwAAAAAAAAAAOZXpbI+gv/dj9qhp1WrVhk6dGjefffd7Lfffpk0aVJ23HHHnH/++bnvvvtSU1OTo48+OkkyZMiQvPPOO7nmmmtSX1+f9u3bZ7vttpsriwAAAAAAAAAAgPnFj9qhJ0n69u2brbfeOrW1tVlwwQVTX1+furpvPuaOO+7Ic889l/Hjx2eJJZbII488kjXWWKPhdQAAAAAAAAAA4P/t/1TatGnT5tsP+P9jna+++ipjxozJ5ptvniWWWCJJsummmyZJSqVSamp+1GZAAAAAAAAAAADQJP1Xlc2XX36ZTz/9NA8//HC22mqrfPzxx+nbt+93v0TMAwAAAAAAAAAAP8j/+SysL774Iv369UuSvPLKK+nXr18GDx48p+YCAAAA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TAUAAAAAAAAAUH0EPcxx5XJ5tt9nHalVU1OTMWPG5MUXX2zYkefII4/MSiutlHPPPTeff/55JcYFAAAAAAAAAKgqgh7mqC+//DJffPFFkqRYLCZJampqMnny5PTr1y+bbLJJdtttt+ywww4ZOnRokuTCCy/MrbfemkcfffQ7MRAAAAAAAAAAQFMj6GGOmThxYnbYYYfstNNOSb4JeZJkxowZOeSQQzJx4sQMHz489913X7bYYovst99+GTNmTLp3757tt98+Z5xxRj744INKLgEAAAAAAAAAoOIEPcwxbdq0yR577JGXX345jz32WAqFQpLkrbfeyvDhw3P11VdnpZVWyquvvprLL788yy67bCZOnJgkufTSS/PKK6/k1ltvzYwZMyq5DAAAAAAAAACYa0quil+NgaCHOaJYLKa2tjZbbLFFNttss/zud79reG3ChAnp2rVr3njjjWy++ebZZ5998utf/zqvvfZaunTpkmnTpqV9+/YZMGBADj/88Lz77ruVWwgAAAAAAAAAQIUJeviv1NfXJ0lqa2uTJB06dMivf/3rvP/++/nzn/+cJCkUCnnkkUfSq1evLL300nnllVdy1FFHpWXLlhk1alQuueSSJMn555+fiy++OCuuuGJlFgMAAAAAAAAAUAUEPfxX6urqkiSXXHJJTjvttHzwwQfp1q1b+vfvn9NOOy3Tpk1L9+7ds+GGG2bZZZfNcccdlyWWWCJJMm7cuFxxxRX58MMP88UXXyRJDjjggEotBQAAAAAAAACgKgh6+K88//zzWXnllXPmmWemWbNmGT9+fFq2bJlf/epXadu2bcPRW+eee26mTp2aXXfdNccdd1wGDhyYrl27ZsyYMTnggAOy8MILV3YhAAAAAAAAAABVoq7SA9B4lEql1NTM3oCdcsop2WijjTJkyJA0a9as4fXVV189BxxwQE466aQMGDAgq622Wq6//vrceOONefvttzNhwoQMGjQoe+65ZyWWAgAAAAAAAABQtQQ9/GCzYp1yuZxCoZARI0bkqaeeyvXXX58WLVo0PFcsFtOiRYtst912ueOOO/L73/8+d999d7p27ZquXbumWCymtra2UssAAAAAAAAAAKhqjtziB3v22WfTp0+fPPTQQ0mSZs2aZdKkSVl11VWTJPX19UnSEOussMIKGTBgQO69997cfvvtDZ8j5gEAAAAAAACgqSqn4Krw1RgIekjyzXFa/65YLH7nmUUWWSQvvfRSHn744UyePDlLLbVUVllllZx99tlJkrq6uobPGjFiRD799NNsttlmOeGEE7LsssvO/UUAAAAAAAAAAMwHBD0k+eY4rcmTJ+e9995L8s0uOhMnTsyIESManllhhRXy61//Ovfff3+eeOKJdOjQIbvssktuuOGGPP744w3PPfroozn//PPzzjvvZLHFFsspp5ySNdZYY56vCQAAAAAAAACgMRL0kCQZO3Zstt9++5x++ukN9/baa6/89re/zWuvvdZw77e//W0WWGCBDBs2LJ999lkGDBiQrbfeOr169comm2yS3r17Z/vtt8+yyy6b9dZbrxJLAQAAAAAAAABo1AQ9TdC/H6dVLpeTJEsvvXTWXnvtvP3227nrrruSJH/4wx8yYcKE3H///Zk2bVqSpFWrVhkwYEAeeuih3HXXXVl44YVz5ZVX5s9//nO6d++eTp065ZVXXslZZ52VQqFxnDsHAAAAAAAAAFBNBD1NUG1tbd544428/fbbs0U3Bx10UGpqanLbbbdl4sSJ6dKlS7bbbrtcf/31+fvf/97wXP/+/dOyZcvcfPPNDff79++fP/7xj/nTn/6U5Zdffp6vCQAAAAAAAABgfiHoaYJuueWWrLbaatlkk01yxBFHZPr06UmSFVdcMdtss03eeOON3HTTTUmSU045JZMnT86tt96azz77LEny9ttvZ4EFFshTTz2Vxx9/PKVSqWJrAQAAAAAAAACY3wh6mqCtttoqrVq1yqKLLpphw4Zlu+22yyWXXJIk2XfffbPkkkvmnnvuyejRo7PooovmsMMOy0MPPZQTTjghb775ZoYMGZL9998/l112WQ455JDU1PhrBAAAAAAAAAAwpygxmpj6+vostNBCOfnkk/P555/nT3/6U7p06ZLf/va3Oeigg/LFF1/k4IMPzieffJLrr78+SfKb3/wmffv2zYgRI7LZZpvlqaeeSu/evdO3b9+0aNGiwisCAAAAAAAAgMaj5Kr41RgIepqY2traJMmRRx6ZmpqaPPTQQzn99NNz00035YMPPsg666yTf/3rX1lmmWXy4osvZtSoUUmSY445Jg8//HDuv//+PP/88+nUqVMllwEAAAAAAAAAMN+qq/QAzB3lcjmFQuE79wuFQurr61NXV5eLLrooffr0Se/evdOzZ8/07NkzRx11VP72t7/l8ccfz6KLLpp27dplvfXWS7NmzbL44otn8cUXr8BqZjd9+vRMnz59tnszy8U0K9RWaCIAAAAAAAAAgDnHDj3zmWKx+L0xT7lcbvi5ru6bjqtXr17p1q1bBg0alA8++CBJcsYZZ+SGG27IDjvskMmTJ2exxRZLqVRdG04NGjQobdu2ne2678vXKz0WAAAAAAAAAMAcIeiZj5TL5dTW1qZQKOSpp57KZZddlkceeSRJvhP4FIvFJMkll1ySJ554Ig888EDq6+tTW1ub9u3bZ+jQoXnvvfdy5plnNgRA1eLYY4/NpEmTZru2ab1qpccCAAAAAAAAAJgjBD3zkUKhkM8//zx9+vTJDjvskFtuuSU77bRTDjjggLz22mtJvt2pp7a2NsViMT//+c9z4IEH5uSTT87o0aMbPmuRRRbJkksuWZF1/G9atGiRNm3azHY5bgsAAAAAAAAAmF8Iehqxfz9Ga5a//OUvmTRpUt54443cf//9eeSRRzJ06NBceeWVmT59+mw79cz6efDgwZk4cWLGjx8/z2YHAAAAAAAAAOD7VddZSvwgs0KeWUHOOeeck2bNmuWwww7LrbfemoEDB6Z9+/a59NJLc8YZZ6Rr167Zdddd06JFi9k+p6amJvX19WnevHk+/vjjLLDAAvN8LQAAAAAAAADQlJQqPQCNgh16GqFCoZBCoZCPPvoot912W84777zU1X3TZn3++ed54YUXsskmm+TUU0/NsccemxEjRqRLly758ssvM3PmzNk+a9b7xDwAAAAAAAAAANVB0NNI3Xzzzdlwww1zwQUX5Oqrr87BBx+cqVOnZosttsgJJ5yQrl275s0338z++++fmpqavPzyyxk8eHAmTJhQ6dEBAAAAAAAAAPh/EPRUuWKx+L33W7dunY4dO+btt99Ojx49kiStWrXKpptumuWWWy6LLbZYFlpooSTJ+PHjc95552XUqFHf2aEHAAAAAAAAAIDqIuipUuVyOaVSKbW1tUmSF198MW+88UamTZuWJNlss83Sq1evfPbZZ3nkkUca3rfddtvlkEMOycknn5x11lknu+22W1ZZZZV8+OGHufjii7P00ktXZD0AAAAAAAAAAPwwdZUegG/inUKhMNu9QqGQQqGQ1157Lb/+9a8zYcKETJs2LZtuumkOP/zwrLXWWtlyyy3zxBNP5Nxzz23YpadNmzY59NBDs9JKK+Uf//hH3n333Vx33XXZZpttKrE0AAAAAAAAAAB+JDv0VFCpVEqS78Q8s47Zuuuuu9KvX7+sueaaefzxx3P11VdnypQpOfroo5Mka6yxRnbYYYe89957+etf/5okDUdqbbnlljnkkENy7rnninkAAAAAAAAAABoRQU8FlMvlJElNzTd//BdffHEOPvjgnH/++fn6668bjtkqlUrZZZddctFFF2XJJZfMiy++mIcffjgPP/xwLrrooiRJjx49stFGG+WSSy7J5MmT06xZs8osCgAAAAAAAACAOULQUwGzduS5/fbb06FDhwwZMiSff/55Bg4cmH79+jU8t8MOO+Swww7L22+/nQ022CCXX355zjvvvOy333457bTTMm3atCyzzDLZYost8sknn+TWW2+t1JIAAAAAAAAAgB+gnIKrwldjUFfpAZqiL774IgceeGCGDRuWiy66KPvvv3+KxWIeffTRbLPNNnn33XezzDLLJEnatGmTP/7xj1lmmWVy1llnpWPHjvn444/z0Ucf5fjjj8+5556bLbfcMquuumpWXnnlyi4MAAAAAAAAAID/mh16KuD999/P6NGj07dv3xx44IGpra1N8+bN8+WXX2bbbbdN69atG56dOnVqbr755qy44orp2LFjkuTzzz/PBhtskKuuuioTJkxImzZtxDwAAAAAAAAAAPMJQU8FrL766tljjz3y3nvv5fbbb0+SDBkyJHvssUdeffXVrLnmmjn44IPz1ltvpVWrVtlss81y/fXX58EHH8zRRx+dUaNG5Zxzzsl7772Xdu3aVXYxAAAAAAAAAADMUY7cqpC+fftm5MiROeuss/L73/8+SfLnP/856667bp5++ukcdNBBqaury5/+9KccfPDB+fjjj/Ob3/wmLVq0yGWXXZZf/OIXFV4BAAAAAAAAAABzg6CnQpZaaqnstNNO+cMf/pD27dvnscceS01NTcrlclZdddXceuutGTlyZKZNm5Z11lkn99xzT955552ssMIKlR4dAAAAAAAAAIC5yJFbFdS7d+907949tbW1efXVV5MkM2fOTLlczgILLJAOHTqkru6b5qqmpkbMAwAAAAAAAADQBAh6KqhFixbp169fyuVyLr/88iRJM1p03wAAx1VJREFU8+bNM2jQoIwaNSr9+/dvCHoAAAAA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AAAAAAAAgOog6GGueeWVV7L22msnSerr61MsFvOnP/0p/fr1y9NPP5177rknZ599dq677rrMmDEj5513XoUnBgAAAAAAAACoPEduMUeUy+UUCoXZfu7fv3/Dvfr6+rRs2TKFQiEffPBB6uvrs9JKKyVJNttss7z44ou57rrrcvPNN2fnnXeu2DoAAAAAAAAAACrNDj3MEbPCnYkTJ8527/PPP89WW22Vww8/PEmy1157ZdNNN83HH3+cf/zjHw3P9u7dO8svv3z++te/ZurUqfN2eAAAAAAAAACAKiLoYY7p3bt3jjzyyBQKhYwaNSonnXRSFl100ay++uoZMWJEXn311bRp0yY77rhjllpqqVx++eUN711uueXSq1evvPvuuzn77LMruAoAAAAAAAAAgMoS9PBfK5fLSZL99tsvf/vb37LRRhtlk002yfTp05Mkv/zlL/OTn/wkZ5xxRpJku+22y3rrrZcRI0bkkUceaficrbfeOltvvXXWWmuteb8IAAAAAAAAAIAqIejh/2xWyDPruK1JkyZl+vTpeeeddzJq1KiGgGfddddNr1698vzzz+fOO+9MkvTt2zeLLLJIrrrqqhSLxSRJu3btcs4552T77bevwGoAAAAAAAAAYO4ruyp+NQaCHn60UqmUUqnUEPLMssIKK+TSSy/Nxx9/nLfffjszZ85M8k3ws80226Rz584ZPHhwkmSdddbJeuutl2eeeSaPP/54w2f8z88EAAAAAAAAAGhqBD38KKVSKTU1Nampqcmbb76ZCy+8MMOHD8+MGTOy7rrrZr/99ssuu+ySU045JWPHjm143worrJDevXvniy++yDnnnJPkmyO6brzxxmy22WaVWg4AAAAAAAAAQNUR9PCj1NTUZPr06TnooIPyi1/8Itddd11222237LjjjnnnnXeSJJdccknGjBmT66+/PtOnT29475ZbbpmVV145t9xyS77++usstdRS6dy5c6WWAgAAAAAAAABQlQQ9/D+Vy7OfHjd16tSccsopefPNNzNixIg8/fTTeeONN/LYY4/lyiuvzPjx49O6dev84Q9/yAUXXJAnnngi06dPz0knnZQpU6bkzDPPzMMPP5wFFligQisCAAAAAAAAAKhugh6+V7lcTrFYTKFQ+M5ra6+9di688MKsttpqufvuu7PlllumUCjk+uuvzzPPPJMkOe6447LMMsvk6KOPzpJLLpnrr78+9fX16dixo5gHAAAAAAAAAOD/QdDD9yoUCqmtrc17772Xq666KmPHjk19fX1atWqV7t27Z/XVV8/pp5+eQw45JL17985nn32Wcrmca6+9tuHorVtuuSUnn3xyrrnmmowePTrLLrtshVcFAAAAAAAAAFD96io9ANWjWCymtra24ffzzz8/J5xwQtq1a5eFFlooe+21V4466qgsuuii+fjjj3PHHXfkuOOOy/77758kWX755TNy5MjcfffdOeSQQ/LTn/40P/3pTyu1HAAAAAAAAACARknQQ5JvjtiaFfNMnDgxrVu3zr/+9a88+uijWXLJJXPRRRflvPPOS48ePbL22mtn5MiRGTt2bFZZZZUkyb/+9a8svvjiWWqppTJjxoyUy+VKLgcAAAAAAAAAqlKpUOkJaAwcudXEFYvFJN8csfXUU09lrbXWynbbbZe99947o0ePzsorr5yf/vSnOeCAA7L22mvnsMMOS5L06dMnNTU1Oeqoo3LYYYdlk002yXLLLZdHH300Rx111Gw7/QAAAAAAAAAA8MMJepq42trafP755/nnP/+Z448/Pttvv3023HDDjBw5MvX19WndunWSpFOnTvn1r3+dN954I9dcc02S5KqrrsovfvGLvPDCCznuuONy8sknZ6GFFqrkcgAAAAAAAAAAGj1HbjUx5XI5hcK3+3fNnDkzXbt2zYwZM7LzzjvnlFNOyfTp07PBBhtkp512ypNPPplu3bqlUChkww03TL9+/XLSSSdl9913z+abb57NN9+8gqsBAAAAAAAAAJj/2KGniSiVSimVSrPFPEnSrFmznHnmmRk3blzat2+fJGnRokU233zz9O7du+GIrSRp165d+vXrly+++CI33njjPJ0fAAAAAAAAAKCpEPQ0ETU1NampqckLL7yQgQMH5rrrrsvkyZOTJL/85S+z0UYb5ZFHHsm0adOSJK1bt85xxx2Xt99+O3/9618bPmfdddfNSy+9lH79+lVkHQAAAAAAAAAA8ztBz3ysVCo1/DxjxowMGDAgG264YZ588skcfPDB2WabbXLdddclSc4444w88sgjuffeexve07lz5xxwwAHZd999M3369CRJq1atsvTSS8/bhQAAAAAAAAAANCF1lR6Auaem5tte68knn8xjjz2WJ598Ml27ds2HH36YI444IkOGDEnnzp2zwQYbZLfddssJJ5yQzTbbLAsvvHCaNWuWAQMGpG3btqmr81cFAAAAAAAAAP5bpf/9EbBDz/ymWCw2/HzrrbemV69eefXVV/PCCy8kSVZaaaUkSYcOHfKb3/wmrVu3ztChQ5N8s0vP+PHjc8EFFzR8xrLLLpsTTzwxtbW183AVAAAAAAAAAABNl6BnPlNbW5sPP/wwr7/+egYNGpRu3bpllVVWyUcffZTFFltstuCnW7duWXrppTN27NhMmzYtSy21VAYMGJC77rorU6dOreAqAAAAAAAAAACaLkFPI1Yul1Mqzb4Z16RJk7Lccsuld+/e2WqrrXLUUUeltrY2u+++e0aMGJFRo0bN9nzz5s3z9ttvp2XLlkmSk08+OX//+9/TqlWrebYOAAAAAAAAAAC+JehpZO699968+uqrSZJSqZSamtn/EbZt2zZnnnlm3nnnnSy11FJJvgl/1lprrfTt2zdHHnlkHnvssZRKpXz88cf5xz/+kV133bXh/Y7WAgAAAAAAAACoLEFPI/Lee++lT58+uemmm1IsFlNbW5v7778/hx56aC688MJ8/fXXSZIDDjggK6ywQkaOHJn6+vqG919xxRVZfPHF07dv32y99dbp3LlzpkyZMlvQAwAAAAAAAABAZQl6qti/H6dVKpXSqVOnHHHEEbnjjjvy2GOP5cwzz8xuu+2W999/P0cddVT23nvvvPXWW2nZsmUGDhyYG2+8Mc8991wKhUKKxWIWXHDB3HTTTbnuuuuy2WabZciQIXn66afTqVOnCq4SAAAAAAAAAIB/J+ipQrNCnlnHaT366KN58cUXkyQnnnhiZsyYkdtvvz1vvvlmHnnkkdx222159NFH8/e//z033nhjvv766/zyl79Mt27dcsIJJ2TGjBkNR2m1a9cuW2yxRY455pjsvPPOlVkgAAAAAAAAAAD/kaCnCs0KeS699NL8/Oc/z5lnnpmddtopjz76aBZYYIEcc8wxueiii/L6669n1VVXTZKsv/762WmnnXLvvffmiSeeSJKcc845efLJJ3P99ddXbC1zw/Tp0zN58uTZrvpysdJjAQAAAAAAAMD/quSq+NUYCHqq0GeffZZdd901p5xySo477rhce+21eeihh9K9e/ckyZ577pktttgitbW1eeeddxre9/vf/z4zZ87MXXfdlY8//jhrr712evfunTvvvDPF4vwTvAwaNCht27ad7Xpi0uuVHgsAAAAAAAAAYI4Q9FShhx9+OK+99lpuvfXW7LnnnmnXrl2WX375JEl9fX2S5Ljjjsv777+fhx9+uOFeu3btcsABB+Shhx7K7bffniS5+uqrc8sttzQcuTU/OPbYYzNp0qTZro3brlrpsQAAAAAAAAAA5ghBT5UpFos544wz0rlz56y77rrfeb2uri5JsvHGG6dHjx7529/+lpdeeqnh9QMOOCAdOnTIQgstlCRp3rz5PJl7XmrRokXatGkz21VXmH+CJQAAAAAAAACgaRP0VFC5XP7O79OmTcuXX36ZpZZaKklSKpW+88ysHXkGDhyYjz76KHfddVcmTZrU8My9996b3XbbLUlSKBTm5hIAAAAAAAAAAJjDBD3z2OjRo3PfffelVCp9J7YpFAoplUopl8v56KOPMnHixNTU1MwW/hQKhdTV1WXChAlZeuml069fv/z5z3/O66+/3vBMy5Yt59l6AAAAAAAAAACYswQ989jvfve7HH/88Xn11Ve/81qpVErr1q2z/fbb5957783zzz+f5Lu77Fx66aXZa6+9kiSnnnpqLrroomywwQZzfXYAAAAAAAAAAOY+Qc88UiwWkyRnnXVWJk6cmHvuuSdTpkxJ8u3RW7PCnaOPPjrlcjlDhgzJa6+9NtvnjBs3Lk8//XR69OiRYrGY2tra9O3bdx6uBAAAAAAAAAD4vyq7Kn41BoKeeeDfj8xaeeWVs/POO+e2227LM888k+TbkKdQKKRcLmfJJZfMBRdckEcffTR9+/bN1VdfnVtuuSWnn356unbtmi+++CK77rpramtrK7IeAAAAAAAAAADmnrpKD9AUFAqF1NbW5rXXXsuVV16ZxRdfPP/85z9z5513Zo011ki7du1SLpdTKBQa4p7+/fundevWueSSS3LUUUelY8eOKZfLOf/889OvX78KrwgAAAAAAAAAgLnFDj3zyLBhw7L++utn8uTJmTFjRlZYYYVcccUVefLJJ5N8u0tP8u2OPjvssEPuueeejB07NsOGDcsLL7wg5gEAAAAAAAAAmM8JeuawUqn0vb8PHz48m266aS699NKceOKJefbZZ7PccsvlyiuvzL/+9a8k34Y8/zPuadGiRZZbbrl5tAIAAAAAAAAAACpJ0DMHFYvF1NR880c6ffr0JElNTU1mzJiRZ555Juuvv36SZMaMGUmSs88+O8OHD88DDzyQYrE4W8gzy6zPAwAAAAAAAACgaVCLzAHFYjFJUltbm/fffz977rln9txzz9x666357LPP0rx586y00kq59957G55Lki222CLLL798brrppjz//PMVmx8AAAAAAAAAgOoh6JkDZgU6L730UnbcccdMmTIlU6ZMyVFHHZULL7wwSbLPPvvkueeey913393w/NixYzNz5sw8+uijue+++xp27gEAAAAAAAAAoOmqq/QA84O33347J5xwQj777LP07NkzJ510UkqlUk466aTcdddd6datW7beeuvsueee2WOPPXL22Wdnww03zI033phddtklyy67bLbZZps0b9680ksBAAAAAAAAAOaiUqHSE9AYCHp+hHK5nFKp1LDDzixt2rTJjBkz8swzz+Q3v/lNkqSmpib9+vXLq6++miFDhqR79+655JJL8vXXX+fss8/OsccemwUWWCC33HJLunTpUonlAAAAAAAAAABQhRy59QMVi8UUCoXU1tZm8uTJef311zN9+vSUy+W0b98+++67b5Zccsk8/vjjDe9ZbbXVsu2222bs2LG57LLLkiSXX355nnrqqdx+++159913xTwAAAAAAAAAAMxG0PMfPPnkk7P9PmtXnhNPPDErrbRSdtlll2yxxRa55ZZbkiRbb711tt5667zwwgt5+umnG97Xq1evrLnmmrnwwgszfvz4tGjRIu3atcsGG2ww7xYDAAAAAAAAAECjIej5HjfccEM22WSTvPrqqw33Pvzww/Tq1SsPPPBALrnkktxyyy3p3LlzzjzzzIwcOTJ1dXXZeeed06xZswwdOrThfUsssUS23nrrbLfddmnRokUllgMAAAAAAAAAQCMi6ElSLpdn+71v375ZZ511MnDgwBSLxSTJAgsskEUXXTQ33XRTtt9++7Rs2TKjRo3KG2+8kTPPPDNJsvHGG2eTTTbJG2+8kRtvvLHh83beeeeceeaZWWSRRebdogAAAAAAAAAAaJSadNBTKpWSJIVCYbb7NTU1GTRoUG666aY8+OCDSZJFFlkkAwcOTKdOnXLsscemS5cu6dq1a0444YS89NJLueaaa5J8EwPV1NTkjjvuyIwZM7738wEAAAAAAAAA4D9pkkHPrB15amq+Wf7FF1+cgw8+OOeff36+/vrrJMlmm22W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pLm5OQ8//HA23HDDjBs3ruR1AAAAAAAAAADVIehhvlhppZXyne98JxMmTMj48eOTJAsttFDWWGONnH/++SWvAwAAAAAAAID5o57Co+RHRyDoYZ4riiJNTU3ZeOONs/322+ewww5LkvTp0ydDhw7NH//4x5x++uklrwQAAAAAAAAAqAZBD/PEnDlz2v/c1NSUJFlqqaXy5S9/OTNmzMjxxx+fJNlss83yX//1X7nooovy+uuvl7IVAAAAAAAAAKBKBD3ME7VaLUly5ZVX5s4778xzzz2XJOnbt2++/OUvZ8yYMXnzzTfTs2fPDB48OF26dMn3v//9MicDAAAAAAAAAFSCoId54s4778yyyy6bo446KgceeGA22GCDjBs3LgsvvHD23HPPLLfcchk5cmSSZMCAARk8eHB+9rOf5eGHHy55OQAAAAAAAABAuQQ9fOLefvvtHHXUUdl7773zu9/9Lvfdd1+GDRuWY489Nrfccks+85nPZMSIERk/fnweeeSRdO3aNVtssUUWXnjhXHXVVWXPBwAAAAAAAAAolaCHj23OnDn/8PnHHnssL7/8crbZZpt06tQpSy+9dE477bQsscQSueGGGzJ9+vQMHjw4AwcOzMEHH5wk2XDDDXPhhRdmzJgx8/EKAAAAAAAAAACqR9DDx1ar1ZIkd999d37729/mrbfeSpK0tLTkzTffzGc+85kkybvvvpuFFloow4cPzy233JJZs2alZ8+e2WOPPfLEE0/koYceSpKss8465VwIAAAAAAAAAMwnhUfpj45A0MO/9dhjj/3D5++4446suuqq+drXvpYhQ4Zkhx12yMSJE7PJJptk5ZVXzvHHH58k6dSpU5Jk3XXXzVtvvZVXX301SbLVVlvlxRdfTP/+/efPhQAAAAAAAAAAdACCHv6lZ599Nv369cull16aJKnX60mSl156KYcddliGDRuWxx9/PD//+c/z7rvvZujQoZk6dWq+/vWv5+yzz859992X2bNnJ0nGjx+fTTfdNKuvvnqSpHv37ll22WXLuTAAAAAAAAAAgIqqlT2AaltxxRUzYsSIHHPMMdljjz3S3PxeA3bVVVela9euOf744zNnzpycdtppefrpp7PbbrtlkUUWyV577ZUJEyZkp512ylprrZWuXbvm7rvvzumnn57W1taSrwoAAAAAAAAAoLqc0MM/9P5JPK2trfna176WGTNm5Nhjj23/eUtLS9Zaa61ceOGFWW655fLQQw/ltttuy/nnn5/OnTuna9euufDCC3PmmWdmo402ymc+85n8/ve/z7777lvWJQEAAAAAAAAAdAhO6GEuRVGkXq+npaUlSTJnzpysttpqGTVqVI444oh85Stfyac//enMmDEjV1xxRW666aacdNJJ2XXXXVOr1TJnzpyceeaZ6dOnTwYOHJidd945O++8c8lXBQAAAAAAAADQcTihh7k0NTWlpaUlkyZNysEHH5zzzjsv7777bvbee++sueaaOfjgg5Mk++yzTxZffPFsu+22GTZsWGq199qwe+65Jz/5yU/y6quvlnkZAAAAAAAAAAAdlqCHDzj11FOzyiqr5OWXX86iiy6aN998M4sttliOOeaYXH/99bnnnnuy/PLL5+tf/3oeeeSRrLfeejnuuOOyyy67ZPvtt88mm2yS4cOHl30ZAAAAAAAAAAAdkltuNbCiKJK8dyrP+37961/n4osvzsUXX/yBW2UNGjQoQ4YMycEHH5wnn3wyBx98cLbYYouMGTMmL774Yjp37pynnnoqK6200ny9DgAAAAAAAADoKOopyp5AByDoaWDvhzzTpk1Ljx49kiQ//elPM3PmzAwYMKD9dfV6Pc3NzenatWuOOOKIbLbZZrnwwguz7777pk+fPhk7dmzmzJnTftstAAAAAAAAAAA+PrfcanBnnXVWhg8fnldeeSVJMmXKlLS2tmaZZZZJvV5PkjQ3//1r0q9fv3zlK1/J/vvvn5kzZ7Y/L+YBAAAAAAAAAPhkCHoaxPu31/q/mpubM2nSpFx33XVJksGDB+fpp5/OL3/5y7lCnqlTp+YnP/lJiqLIwQcfnO9+97siHgAAAAAAAACAeUDQs4B74oknkvz99lqPPPJIfvnLX7b/fP/998/qq6+e22+/Pc8880z69++f7bffPsOHD8/kyZMzY8aMtLW15eyzz864ceMyadKkrLDCCjn66KPT0tJSxiUBAAAAAAAAACzQBD0LsFtuuSU77rhjzj777CTJq6++mr322is/+tGP8uc//znJe7fKGj58eN5+++1cdtllWWihhXLuuedm4YUXTv/+/bPddtulT58+Offcc7PbbrulZ8+eJV4RAAAAAAAAAMCCT9CzAKjX6//w7+uss04GDBiQ66+/Pq+//nqWW265fPnLX85zzz2Xm2++uf31gwYNyvrrr5/bbrstd911V5ZaaqnceuutOfXUU7PhhhtmxIgReeWVV7LlllvO1+sCAAAAAAAAAGhEgp4FQHPze/8z/vSnP82kSZPa/96rV69st912eeedd3LGGWckSUaMGJFu3brl5ptvzosvvtj+GXvttVeef/75/PjHP8706dOz3HLLZccdd8xxxx2XAw88cP5fFAAAAAAAAAAsgOoepT86AkHPAqAoigwbNixf/OIXs+WWW+aHP/xhZsyYkST57//+72y00Ua5/fbb89hjj2WRRRZpj3fGjx/f/hm/+93vsvTSS+eBBx7IfffdV9alAAAAAAAAAAA0PEFPB1ev19PU1JRtt902Cy+8cD796U/n6KOPzrBhw3LfffelW7du2X333bPwwgu3n9Kz2267pV+/frnqqqty3HHH5a677srVV1+dI488MhdffHG22Wabkq8KAAAAAAAAAKBxCXo6oKIo2v/8/u219thjj6y//vrp1atXTj755PTs2TNDhgzJoYcemrXXXjvbbLNNnnnmmVx33XVJkm9961sZNGhQLr300uy+++5ZffXVs88++6Rfv36lXBMAAAAAAAAAAO8R9HQQd999d3bcccckSVNT01w/a2trS5J897vfze23357Zs2fnnHPOyUknnZSbb745X/ziFzN58uSstdZaufzyyzN79uysuuqqOfHEE3P77bfnmWeeyUknnTTfrwkAAAAAAAAAgA8S9HQQc+bMyQ033JDx48cnee9WW+9raWlJkmy++ebZbLPNcumll+bJJ5/MPvvsk3vvvTdrr712rrjiitx44425+eabc8EFF7S/d5VVVsliiy02fy8GAAAAAAAAAIB/StDTQQwYMCAHHHBAvvnNbyb5+6223vd+4HPCCSdk8uTJue666/LWW29lqaWWyimnnJLrrrsuG2ywQbp3754VV1xxfs//RM2cOTNvv/32XI85RVvZswAAAAAAAAAAPhGCngoriqL9z62trfn617+ed955JyeccEKSuU/paW5uTr1eT69evbLPPvtk3LhxmTBhQvvPBwwYkKuvvjpvvvlmtt566/l3EfPA6NGj06NHj7keD097puxZAAAAAAAAAACfCEFPBdXr9dTr9TQ1Nc31/KqrrppRo0blhBNOyBtvvJHm5ua5op/3X3/kkUemS5cuGTt2bF599dX2n3fv3v0DJ/t0RIcffnimTZs21+PzPdYsexYAAAAAAAAA/FuF/5T+n46g49cdC5h6vZ7m5uY0Nzfn2Wefzfjx4/PSSy8lSWq1Wnbfffesssoq7bfe+t+amprS1taWlpaWHHTQQZkyZUo6deo0vy9hnmttbU337t3netSaWsqeBQAAAAAAAADwiRD0VExzc3PeeeedDB8+PJtuummOOOKIDBo0KEcccUSSZNlll80xxxyTq666Kg8//HCamprmOqWnpeW9sGWfffbJz372syy99NKlXAcAAAAAAAAAAB+PoKdiJk2alK985SuZPHly7r333jz55JM5+eSTc+KJJ+auu+5KU1NTvvCFL2Tw4MH5+te/niQfuDVXkrkiHwAAAAAAAAAAOg5BT0nq9Xra2to+8Hxzc3M++9nP5sorr8zqq6+en/3sZ/nOd76TJPn2t7+d6dOnZ5FFFsl3vvOdPP300/nRj36U5IMBzz+KfAAAAAAAAAAAqD5BT0mam5vT0tKSl19+OXfffXdmz56dJFl66aWz1157ZdFFF82IESMyYsSI7LHHHnn00Ufz1FNP5dJLL02S9O3bN1/60pcyduzYJAIeAAAAAAAAAIAFhaBnPnr/RJ6iKFIURQ477LCsueaa2WOPPbLDDjvkgQceSPJe1POrX/0qDz/8cM4///wcfvjhWXLJJbPoootmzJgxefHFF9OpU6eMGTMmDz74YJmXBAAAAAAAAADAJ0zQM58URZGWlpYk752mM2nSpLz00kt58MEHc9lll2Xq1Kk59dRT89prryVJbrvttrz++uvZaqutkiQvvfRSBgwYkD/96U+5//77kyQ9evQo52IAAAAAAAAAgI+l7lH6oyMQ9Mxj75/K09TUlNtvvz0DBgzIsGHDcsopp6RXr17p06dPBg4cmK9//euZNGlSLrjggiTJtttum9deey0HHnhgTjjhhOy9997Zcsst88orr2T48OFlXhIAAAAAAAAAAPOQoGceqdffa7paWlry1ltv5fnnn883vvGNbLDBBvnTn/6UM844I506dWp//ZAhQ/LZz342P//5z/PYY4/lc5/7XE4//fS8+OKLufzyy3P44Ydn3333Tbdu3cq6JAAAAAAAAAAA5gNBzyesKIokSXNzc+bMmZOnnnoq6667bnbfffeMGDEiJ598cq644orst99+ufHGG/Pqq68mSVpbWzNs2LC0tra2n9Jz0EEH5Zprrslvf/vb7LfffqVdEwAAAAAAAAAA84+g5xNUFEWampqSJFdddVU22mijPPTQQxk6dGiefPLJ9O/fP0myzDLLZOjQofnUpz6VE088sf39AwYMyHrrrZd77703jz76aJKke/fu8/9CAAAAAAAAAAAojaDnE9TU1JTf//73ueWWW3LZZZflgAMOyO67756ddtopiy22WG644Yb2126yySbZZpttctddd+Xhhx9uf37EiBG57rrr0q9fvxKuAAAAAAAAAACAsgl6/gP1en2uv8+cOTObbbZZ9t9//6y44orZZ599stBCC6Vv37456KCDctZZZ2XSpElJks6dO2fbbbfNCiuskEMPPbT9M1ZYYYWsvvrq8/U6AAAAAAAAAACoDkHPx9DW1pYkaW6e+x9fa2trTjjhhEyePDmdO3duf36hhRbKjjvumNVWWy0jR45sf75Pnz4ZMmRIhgwZkqIoUhTF/LkAAAAAAAAAAAAqq1b2gCoriiJNTU3t//2+lpaWJMm1116bn//851lxxRWzzTbbpE+fPtlrr70yduzY/O53v8tzzz2X1VZbLUmy2mqr5cADD8w3v/nN3H333dl8882TJPvtt1/75wEAAAAAAAAAC7YiDvvg33NCz7/wvyOe5O8n80ybNi1f/OIX89WvfjVJctZZZ2XPPffM+eefnyQ5/PDD89RTT+Xuu+9uf0+tVsvAgQOzzjrr5Lzzzmv/TDEPAAAAAAAAAAD/m6Dn3zjxxBMzdOjQJH8PfO6999689tpruf/++3PuuefmgQceyFZbbZVDDjkk06ZNyxe+8IVsvPHGufLKK/PrX/+6/bN69eqVyy67LD/+8Y9LuRYAAAAAAAAAAKpP0PM3RfHBI63q9XqWWGKJjB8/Po899liam9/7x/Xoo4/m7bffzuqrr57kvVDnoIMOygorrJBRo0YlSU444YS8/PLLueqqqzJ9+vT2z1x++eXnw9UAAAAAAAAAANBRCXr+5v3Td9566632uKe5uTlDhgzJtttum/3337/9tbVaLb169coLL7zQ/lyvXr2y44475ve//32mTZuWVVddNTvssEPeeustt9UCAAAAAAAAAOBDE/T8zbRp07LTTjtl4403ztlnn93+/OKLL56RI0fmqaeeyiWXXJIk6dOnT15//fXcc8897a9ramrKCy+8kFqtlkUWWSRJcuqpp+aCCy7IwgsvPF+vBQAAAAAAAACAjkvQ8zevvvpqnn322bz44osZOXJk9txzz/z85z9PU1NTNtlkk+y777455JBDkiTbb799+vTpk7Fjx2bs2LGZMWNGnn322bz00ksZNGhQ+4k875/6AwAAAAAAAAAAH5ag52/WWmut7Lffftlyyy2z2267ZYkllsiQIUNy1FFH5Z133snBBx+cRRZZJN/85jeTJMcee2z69++f/fbbL1tssUX69u2bT3/60znggANKvhIAAAAAAAAAoKrqHqU/OgJBz/8ydOjQdO3aNX/6059y7LHH5owzzsj48eOz1VZb5Ze//GUOO+ywnH766Zk0aVJ69+6dU045JQ899FAOPvjg/OIXv8i1116bbt26lX0ZAAAAAAAAAAB0YIKe/2W55ZbLdtttlzfeeCMXXnhhvvzlL+fuu+/Oeuutl8MPPzw/+clPUqvVMmrUqPb3bLDBBtlll13Sr1+/EpcDAAAAAAAAADAvnH322VlppZXSpUuXrL/++rnvvvv+6WvvueeeNDU1feDxu9/97iP9TkHP/7Hjjjumb9++ufHGGzNhwoQsueSSOfvss3PllVcmSWbPnp0rrrgif/zjH0teCgAAAAAAAADAvHT11VfnG9/4Ro488sg8/vjj2XTTTbP11ltn4sSJ//J9zz77bCZNmtT++MxnPvORfq+g5//o0qVLdtlllyTJxRdfnCRpamrKFltskRtuuCHXXHNNXnzxxfTs2bPElQAAAAAAAAAAzGunnnpq9t133+y3335ZY401MmbMmPTq1SvnnHPOv3zfUkstlWWWWab90dLS8pF+r6DnH9h8880zYMCAPPHEE7npppuSJG1tbenUqVOGDh2aFVdcsdyBAAAAAAAAAAB8LDNnzszbb78912PmzJkfeN2sWbMyYcKEbLnllnM9v+WWW+aBBx74l79jvfXWy7LLLpstttgid99990feKOj5J4YNG5bZs2fnpptuSltb20cupQAAAAAAAAAAqJ7Ro0enR48ecz1Gjx79gde9+eabaWtry9JLLz3X80svvXQmT578Dz972WWXzfnnn59x48bl+uuvT+/evbPFFlvk3nvv/Ugbax/p1Q1kzTXXzCmnnJINNthAzAMAAAAAAAAAsIA4/PDDM3LkyLmea21t/aevb2pqmuvv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mTJlSpLkscceyyKLLJJ11lknI0aMyGc+85msueaaeeihhzJ48OAkyZQpU3Lsscdmxx13zJAhQzJ06NAcccQRpV0PAAAAAAAAAEAVuOUWH1tz83s92A9/+MOccMIJWXnlldPa2prdd9896623Xh555JEst9xy2XjjjXP77bdno402SpK88MIL+dWvfpVhw4Zl3XXXzRJLLJFLLrkkiy++eJmXAwAAAAAAAABQCYIePraiKHLKKafk3HPPzemnn54hQ4ZkxowZSZLFFlss+++/f2644YbceOON6dq1a5Lk3Xffzfnnn585c+Zku+22y6677jrX7boAAAAAAAAAYEFWL4qyJ9ABuOUWH8qcOXM+8Fy9Xs+NN96YvffeO8OGDcvs2bNTq9UyZcqU1Gq1jBw5MrNmzcouu+ySE088MePGjcuAAQNy4403ZvDgwVl44YXFPAAAAAAAAAAA/4cTeviXiqJIU1NTarX3vipPPPFEVlpppfTo0SMtLS1ZZpllMm7cuDQ3N+fpp5/OpEmT8vDDD2ePPfbIGWeckdtvvz1nnnlmbrrppsyYMSObb755Tj311JKvCgAAAAAAAACgugQ9/Evvn6AzduzYHHnkkVliiSUya9as7Lvvvhk1alROPPHEHHzwwbn11luz9dZb57/+679yyCGHZNttt80uu+ySzTffPBdffHGmT5+eJOnWrVuZlwMAAAAAAAAAUHmCHuZSFEWKokhzc3Pq9Xqam5szfvz4HHPMMTn22GPTv3//XHrppTn77LMzderUjB49OldffXW6du3afprPO++8k9VWWy2zZs1q/9xFFlnkE7u91syZMzNz5swP7Hb7LgAAAAAAAABgQdBc9gCqYfr06RkzZkyee+65NDc3Z8qUKWlufu/rcfnll6dfv375yle+kj59+uTEE0/MqFGjcvLJJ+eVV15J165d89e//jV/+MMf8qtf/Spbb711evbsmb59+7Z//icZ24wePTo9evSY69HW9vYn9vkAAAAAAAAAAGUS9DSwH/3oR3nggQeSJM8991y+/e1v57bbbsuoUaOy5JJL5tZbb02SdOnSJd27d29/X61Wy3bbbZf11lsvp556apLktttuyyGHHJLBgwdn9dVXz+23354ll1xynuw+/PDDM23atLkeLS3d//0bAQAAAAAAAAA6ALfcaiD/+3Zav/71r3Pbbbdls802S1EUWX/99TNw4MAceuihWXnllfOLX/wim266aZL3gp7XX389zzzzTNZcc80kSc+ePfOpT32q/RSffv36pV6v57TTTssKK6wwT6+jtbU1ra2tcz3ndlsAAAAAAAAAwIJC0NNAmpqa0tTUlFdffTWrrbZaxo0blyT5y1/+klmzZmXatGlZdNFFM2TIkGywwQbt79trr73yla98Jddff3170DNz5sy88cYb2WqrrZIkyy+/fJZffvn5f1EAAAAAAAAA0IEUZQ+gQ3DLrQZSr9dz3HHHZfXVV89pp52WJBkzZkz222+/FEWRBx54IEcffXTGjx+f22+/vf19AwYMyA477JBrrrkmAwcOzEUXXZTtt98+77zzTrbccsuyLgcAAAAAAAAAYIEk6FlAFcUHm77p06fnmmuuyUILLZRf/epXeeWVV9KjR4889dRTufbaa5MkX/va19KjR49cc801mThxYvt7v/Od7+T0009Pa2trxo4dm2WXXTaPPvpo1lprrfl2TQAAAAAAAAAAjUDQswCq1+tpamqa67m2trb06NEjAwYMyKKLLpq2tracf/752XvvvbPmmmvmlltuyeOPP54kOfLII3Pfffflrrvuan//lClTstlmm2X8+PG55ZZbcskll6Rbt27z9boAAAAAAAAAABqBoGcB1NzcnD/84Q/53ve+lwcffDDJeyf2FEWR3r17Z8stt8yyyy6bu+66KxMmTMi3vvWtvPzyy/nJT36SoigyZMiQfP7zn88Pf/jDHHHEEVljjTWy4447Jkm6dOki5AEAAAAAAAAAmIcEPQugqVOn5gtf+EKOOeaYDBs2LBMmTMjs2bPT1NSUd999N5MnT86RRx6ZlpaWnHPOOfnc5z6XgQMH5s4778ydd96ZJPnBD36Q7bbbLr/4xS/ypS99KY8++mjJVwUAAAAAAAAA0BgEPQugxRZbLMOHD8/AgQPTrVu3nHLKKTn++OOTJEOHDs2DDz6YhRdeOLvssksef/zx3HjjjfnWt76V2bNn56c//Wn+/Oc/Z8UVV8wxxxyTu+++O8cdd1zJVwQAAAAAAAAA0DgEPQuor371q+nevXvWXnvtfOlLX8pll12W733ve3n++eez2WabZeLEiRk2bFiWW265XHLJJenRo0eGDh2aG2+8sf00nubm5nTu3LnkKwEAAAAAAACABUc9hUfJj45A0LOAWmyxxbLzzjvnD3/4Q1pbW3P99dfnySefzIknnth+Cs/iiy+eYcOG5cUXX8wZZ5yRr33taxk7dmwGDRpU9nwAAAAAAAAAgIYl6FmA7bTTTllhhRVy1llnZbXVVsvpp5+eVVddNdOnT8/kyZOTJLvuumvWW2+9LLPMMuncuXM233zzklcDAAAAAAAAADQ2Qc8CrFOnTjnkkEMyZcqUnHnmmenZs2d+9KMf5fnnn8+uu+6aonjvGKnzzjsvX/7yl8sdCwAAAAAAAABAEkHPAu9zn/tc+vfvnzvvvDNPPPFEmpqassoqq6Rer6epqSlJ0rlz55JXAgAAAAAAAADwPkHPAq6pqSmHHHJIZs+enVNPPbX9+eZm/9MDAAAAAAAAAFRRrewBzHu9evXKkCFD0qlTpxRF0X4yDwAAAAAAAAAA1SPoaRAjR44U8gAAAAAAAAAAdACCngYh5gEAAAAAAACA8tVTlD2BDqC57AEAAAAAAAAAAMDfCXoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhgh4AAAAAAAAAAKgQQQ8AAAAAAAAAAFSIoAcAAAAAAAAAACqkVvYAAAAAAAAAAIBGURRF2RPoAJzQAwAAAAAAAAAAFSLoAQAAAAAAAACAChH0AAAAAAAAAABAhQh6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACoEEEPAAAAAAAAAABUSK3sAQAAAAAAAAAAjaKeouwJdABO6AEAAAAAAAAAgAoR9AAAAAAAAAAAQIUIegAAAAAAAAAAoEIEPQAAAAAAAAAAUCGCHgAAAAAAAAAAqBBBDwAAAAAAAAAAVEit7AEAAAAAAAAAAI2iSFH2BDoAJ/QAAAAAAAAAAECFCHoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhgh4AAAAAAAAAAKgQQQ8AAAAAAAAAAFSIoAcAAAAAAAAAACpE0AMAAAAAAAAAABVSK3sAAAAAAAAAAECjKIqi7Al0AE7oAQAAAAAAAACAChH0AAAAAAAAAABAhQh6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACoEEEPAAAAAAAAAABUSK3sAQAAAAAAAAAAjaKeouwJdABO6AEAAAAAAAAAgAoR9AAAAAAAAAAAQIUIegAAAAAAAAAAoEIEPQAAAAAAAAAAUCGCHgAAAAAAAAAAqBBBDwAAAAAAAAAAVIigBwAAAAAAAAAAKqRW9gAAAAAAAAAAgEZRFEXZE+gAnNADAAAAAAAAAAAVIugBAAAAAAAAAIAKEfQAAAAAAAAAAECFCHoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhgh4AAAAAAAAAAKiQWtkDAAAAAAAAAAAaRT1F2RPoAJzQAwAAAAAAAAAAFSLoAQAAAAAAAACAChH0AAAAAAAAAABAhQh6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACoEEEPAAAAAAAAAABUSK3sAQAAAAAAAAAAjaJIUfYEOgAn9AAAAAAAAAAAQIUIegAAAAAAAAAAoEIEPQAAAAAAAAAAUCGCHgAAAAAAAAAAqBBBDwAAAAAAAAAAVIigBwAAAAAAAAAAKqRW9gAAAAAAAAAAgEZRL4qyJ9ABOKEHAAAAAAAAAAAqRNADAAAAAAAAAAAVIugBAAAAAAAAAIAKEfQAAAAAAAAAAECFCHoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhgh4AAAAAAAAAAKiQWtkD4JPw7JqrlT0B5rklBncvewIAn4BXr51e9gSY52bNLMqeAPPcZUcMKHsCzHMzH/9j2RNgnnv75c5lT4B5boO19yx7Asxzjzx1WdkTAD6SIv7/M/49J/QAAAAAAAAAAECFCHoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhgh4AAAAAAAAAAKgQQQ8AAAAAAAAAAFSIoAcAAAAAAAAAACqkVvYAAAAAAAAAAIBGUS+KsifQATihBwAAAAAAAAAAKkTQAwAAAAAAAAAAFSLoAQAAAAAAAACAChH0AAAAAAAAAABAhQh6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACokFrZAwAAAAAAAAAAGkWRouwJdABO6AEAAAAAAAAAgAoR9AAAAAAAAAAAQIUIegAAAAAAAAAAoEIEPQAAAAAAAAAAUCGCHgAAAAAAAAAAqBBBDwAAAAAAAAAAVIigBwAAAAAAAAAAKqRW9gAAAAAAAAAAgEZRL4qyJ9ABOKEHAAAAAAAAAAAqRNADAAAAAAAAAAAVIugBAAAAAAAAAIAKEfQAAAAAAAAAAECFCHoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhtbIHAAAAAAAAAAA0iiJF2RPoAJzQAwAAAAAAAAAAFSLoAQAAAAAAAACAChH0AAAAAAAAAABAhQh6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACoEEEPAAAAAAAAAAD8E2effXZWWmmldOnSJeuvv37uu+++D/W+X/7yl6nVall33XU/8u+sfeR3AAAAAAAAAADwsdSLouwJfARXX311vvGNb+Tss8/OxhtvnPPOOy9bb711nnnmmSy//PL/9H3Tpk3L8OHDs8UWW+T111//yL/XCT0AAAAAAAAAAPAPnHrqqdl3332z3377ZY011siYMWPSq1evnHPOOf/yff/v//2/7Lbbbtlwww0/1u8V9AAAAAAAAAAA0DBmzpyZt99+e67HzJkzP/C6WbNmZcKECdlyyy3nen7LLbfMAw888E8//6KLLsoLL7yQY4455mNvFPQAAAAAAAAAANAwRo8enR49esz1GD169Ade9+abb6atrS1LL730XM8vvfTSmTx58j/87Oeffz6HHXZYrrjiitRqtY+98eO/EwAAAAAAAAAAOpjDDz88I0eOnOu51tbWf/r6pqamuf5eFMUHnkuStra27Lbbbjn22GOz2mqr/UcbBT0AAAAAAAAAADSM1tbWfxnwvG+JJZZIS0vLB07jeeONNz5wak+STJ8+PY8++mgef/zxHHTQQUmSer2eoihSq9Xys5/9LAMHDvxQG91yCwAAAAAAAAAA/o/OnTtn/fXXzx133DHX83fccUc22mijD7y+e/fu+c1vfpMnnnii/XHAAQekd+/eeeKJJ/L5z3/+Q/9uJ/QAAAAAAAAAAMwnRYqyJ/ARjBw5MnvuuWf69euXDTfcMOeff34mTpyYAw44IMl7t+967bXXcumll6a5uTlrr732XO9faqml0qVLlw88/+8IegAAAAAAAAAA4B/YZZddMmXKlBx33HGZNGlS1l577dxyyy1ZYYUVkiSTJk3KxIkTP/Hf21QUhfSLDu/lvoPKngDz3BKDP1X2BAA+Aa9eO73sCTDPzZrp3x1hwbfyXouUPQHmuZmP/7HsCTDPvf1y57InwDy3w+S/lj0B5rlHnrqs7Akwz3VaYuWyJ/AJWnmJ9cqe0PBefPPxsif8W81lDwAAAAAAAAAAAP5O0AMAAAAAAAAAABUi6AEAAAAAAAAAgAoR9AAAAAAAAAAAQIUIegAAAAAAAAAAoEJqZQ8AAAAAAAAAAGgURVEvewIdgBN6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACoEEEPAAAAAAAAAABUiKAHAAAAAAAAAAAqRNADAAAAAAAAAAAVUit7AAAAAAAAAABAo6inKHsCHYATegAAAAAAAAAAoEIEPQAAAAAAAAAAUCGCHgAAAAAAAAAAqBBBDwAAAAAAAAAAVIigBwAAAAAAAAAAKkTQAwAAAAAAAAAAFSLoAQAAAAAAAACACqmVPQAAAAAAAAAAoFEURVH2BDoAJ/QAAAAAAAAAAECFCHoAAAAAAAAAAKBCBD0AAAAAAAAAAFAhgh4AAAAAAAAAAKgQQQ8AAAAAAAAAAFSIoAcAAAAAAAAAACqkVvYAAAAAAAAAAIBGUU9R9gQ6ACf0AAAAAAAAAABAhQh6AAAAAAAAAACgQgQ9AAAAAAAAAABQIYIeAAAAAAAAAACoEEEPAAAAAAAAAABUiKAHAAAAAAAAAAAqRNBDKYqiSL1eL3sGAAAAAAAAAEDl1MoeQOOZM2dOarVampqaMnXq1MyZMydLLrlk2bMAAAAAAAAAYJ4riqLsCXQATuhhvqvV3uvIDj300PTt2zfXX399pk6dWvIqAAAAAAAAAIBqcEIP891TTz2VL33pS1lsscVy6qmnZvXVV0+3bt3KngUAAAAAAAAAUAmCHuaZ948Ja2pqmuv5+++/P7169cott9zSfloPAAAAAAAAAADvUVMwT8yZM6c91mlra0tLS0v7z+6///68/fbbefLJJ/PYY4/lT3/6Ux566KFsvvnm2XnnnfPpT386RVF8IAQCAAAAAAAAAGgEzWUPYMHQ1taW5O+n8rwf83z/+9/P/vvvn5NOOikTJkxIkhxxxBGZPXt2Ntlkk1x77bV57rnnUqvV8uMf/zinn356kg+e6gMAAAAAAAAA0Cic0MPHVhRFXnrppey0004566yzstFGG7WHOPfcc0/233//LLTQQhk0aFDGjRuXG264IRdddFHWXHPNXHfddanX61lyySXTqVOnLLTQQtl0003TqVOn9s8W9QAAAAAAAAAAjUjQw0dWr9fT3NycpqamrLzyynnllVdy3nnnZe2110737t3z1ltv5YILLsjw4cNz1FFHJUnGjx+fPffcM0cffXR+/OMfZ6WVVkqSzJw5M62trbn55pszderU9O3bN4kTegAAAAAAAABYMNX/ducb+FfccosPrV6vJ0mam5szY8aM3H///fnrX/+aW2+9NZdddlnuvffe1Ov1fOpTn8pOO+2UAw44IO+8806++tWvZvjw4dlkk01y88035+abb06S/OY3v8nBBx+crbfeOsOGDcvee++dHXfcscxLBAAAAAAAAAAonRN6+NCam9/rv77//e/n4osvziqrrJKnn346v//97zNo0KD84Ac/yLrrrpvlllsuO+ywQ2bPnp1dd901U6dOzf33359lllkm22yzTY4//vhsu+22WWeddbLIIoukb9++ueaaa9KtW7cPtWPmzJmZOXPm3M/V62lt1qcBAAAAAAAAAB2foIcP7cUXX8x+++2XiRMn5qSTTkrfvn3zzjvvpKWlJeeee25WXXXV/OQnP8n++++fWq2WX/ziF3nwwQdz6623pk+fPvnzn/+cOXPm5PHHH8/RRx+d4447Lv/zP/+Tzp07f6Qdo0ePzrHHHjvXcwcvs1K+uezKn+TlAgAAAAAAAACUQtDDh3bVVVdl1qxZueOOO7LSSiu1Pz979uysvPLKGTFiRH7wgx9ks802yxprrJFOnTrl3XffzV/+8pckyR133JG+fftm//33zyqrrJIkHznmSZLDDz88I0eOnOu5yQOGfPwLAwAAAAAAAACoEPco4kN544038oMf/CDbb7/9XDFPkrS0tCRJfvjDH2bq1Km56KKL0tbWluWWWy4bb7xxtt1222y88cbZZ599MnDgwBx00EHZeuutP/aW1tbWdO/efa6H220BAAAAAAAAAAsKJ/Twobzxxhtpa2vLiiuumCSp1+tp/ltE09zcnFmzZqVz584ZPXp0Dj300Oy00075/Oc/nwsuuCBXX311pkyZkvHjx2eppZYq8SoAAAAAAAAAAKpP0MOH8v7tsyZNmpQ5c+akVpv7q/P+rbO++tWv5tJLL81RRx2VK664IksttVS+9rWvlTEZAAAAAAAAAKBDEvTwofTu3TubbrpprrzyygwZMiTLL798iqJIU1NT+2tGjRqVxRdfPKeddlp22GGHzJw5s8TFAAAAAAAAAFA9RYqyJ9ABNJc9gI5j5MiRefTRRzN27Nj86U9/SlNTU+r1epLk6aefzh/+8If07t07/fv3z6RJk7LccsuVvBgAAAAAAAAAoONxQg8f2he/+MV84xvfyAknnJDHH388X/3qV1Or1XLvvffm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V1S8eHHt3r1bS5cuVZMmTQyXAlnn7e2tn3/+WRUqVJC3t7d++uknVapUST///LM6d+6svXv3mk4EssW5c+fk7e1tOgPIslq1aumFF15Q165dJf33i4LmzZtr1qxZqlSpknr16qWgoCB9/vnnhkuBrDt//rzCw8P166+/6v7775ck/fHHH3rssce0ePFi/l2HU6hbt67Kly+vjz/+WN7e3oqNjZWrq6s6dOigvn37qlWrVqYTgSx7/PHH9fTTT6t3797y8PDQjh075O/vr169eunAgQNauXKl6UTAIcqXL68xY8aoadOm8vf317x581SvXj3FxsbqiSee0KlTp0wnAncsIiJCYWFh6t+/f4b7U6ZM0bp167RkyZIcLgMA58cjtwDctrJly6ps2bKmMwCH2rRpk6ZOnZr6evbs2bpx44bi4uLk5eWlwYMHa8KECQz0wCm4urqm3gFZtGhRHTlyRJUqVZKXl5eOHDliuA5wjHfeeUdlypRR27ZtJUmtW7fWokWLVLx4ca1YsUIPPPCA4ULgzu3fv181atRIfb106VJFRESoffv2kqRx48alDvsAuZ2Xl5c2b96sH374QbGxscqfP7+qVq2qOnXqmE4DHGb79u365JNP5OLiIhcXF129elUBAQEaP368OnfuzEAPnMJbb72l8PBw7d69Wzdu3ND777+vXbt26aefflJUVJTpPMBh+vXrp/bt28vd3V1+fn6qW7eupP8+iis4ONhsHJBFsbGxeuedd26637BhQ07/BoBswkAPgEwlJydr1qxZWrNmjU6ePJnumalr1641VAZk3bFjx9KcNrVmzRo99dRT8vLykiR17txZM2fONJUHOFRISIh+/fVXlS9fXmFhYRo+fLhOnTqlOXPm8OESnMYnn3yiuXPnSpJ++OEHrV69WitXrtT8+fP16quvatWqVYYLgTuXlJQkT0/P1NebN29Wt27dUl8HBATo+PHjJtKAbGFZlho2bKiGDRuaTgGyBQP3uBfUrl1bmzZt0rvvvquyZctq1apVql69un766Sf+DoVT6dmzp2rWrKmjR4+qQYMGstlskv77Hn3MmDGG64CsOXHihFxdXW+6nydPHv311185WAQA9w4GegBkqm/fvpo1a5aaNm2qKlWq8HxrOJV8+fIpKSkp9XV0dLQmTJiQZv/SpUsm0gCHGzdunC5evChJGj16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",
      "text/plain": [
       "<Figure size 3000x2500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Multivariate visualisation\n",
    "plt.figure.figsize=(10,8)\n",
    "plt.title(\"Correlation of attribution with the class variable\")\n",
    "a=sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')\n",
    "a.set_xticklabels(a.get_xticklabels(), rotation=90)\n",
    "a.set_yticklabels(a.get_yticklabels(), rotation=30)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1973e46d-6269-4574-92d0-af62307fcb7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Declaring our feature vector and target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e9e60b61-6b5e-4b63-aa6d-101b3f12c11b",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=newdf.drop('Clas', axis=1)\n",
    "y=newdf['Clas']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e5c04bd6-9d81-4cc9-9f82-84f7201de47f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting data into training and test set\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_Test= train_test_split(x,y, test_size= 0.2, random_state= 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5d80184f-0f0d-4f95-9ac6-a9f9841f69c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((559, 9), (140, 9))"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape, x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "99b0d3de-886b-43f1-80be-7132063d7465",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clump Thickness                  int64\n",
       "Uniformity of Cell Size          int64\n",
       "Uniformity of Cell Shape         int64\n",
       "Marginal Adhension               int64\n",
       "Single Epithelial Cell Size      int64\n",
       "Bare Nuclei                    float64\n",
       "Bland Chromatin                  int64\n",
       "Normal Nucleoli                  int64\n",
       "Mitoses                          int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Feature Engineering\n",
    "##Checking data type of the training\n",
    "x_train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "22b48289-15a1-4ffb-bfb2-fc0941b3d280",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clump Thickness                 0\n",
       "Uniformity of Cell Size         0\n",
       "Uniformity of Cell Shape        0\n",
       "Marginal Adhension              0\n",
       "Single Epithelial Cell Size     0\n",
       "Bare Nuclei                    13\n",
       "Bland Chromatin                 0\n",
       "Normal Nucleoli                 0\n",
       "Mitoses                         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###Checking for null values in the training dataset\n",
    "x_train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f9f70f97-30af-4288-bba5-80e549c41465",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clump Thickness                0\n",
       "Uniformity of Cell Size        0\n",
       "Uniformity of Cell Shape       0\n",
       "Marginal Adhension             0\n",
       "Single Epithelial Cell Size    0\n",
       "Bare Nuclei                    3\n",
       "Bland Chromatin                0\n",
       "Normal Nucleoli                0\n",
       "Mitoses                        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###checking for null values in the test dataset\n",
    "x_test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "007f7986-b0e6-4cf3-bdf5-04da7a1805b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bare Nuclei 0.0233\n"
     ]
    }
   ],
   "source": [
    "##Calculating the proportion of missing values in the training set\n",
    "for col in x_train.columns:\n",
    "    if x_train[col].isnull().mean()>0:\n",
    "        print(col, round(x_train[col].isnull().mean(), 4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e477a1b4-2d29-4e9f-aaf3-b8c35e8a5d3f",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Replacing missing values with the median values in the training set and propagating the same value to the test set. \n",
    "###The median is preferred because it's robust and not susceptible to outliers \n",
    "for df1 in [x_train, x_test]:\n",
    "    for col in x_train.columns:\n",
    "        col_median = x_train[col].median()\n",
    "        df1[col].fillna(col_median, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "217a1b99-28e3-4935-baa0-0c52c8a58ec2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clump Thickness                0\n",
       "Uniformity of Cell Size        0\n",
       "Uniformity of Cell Shape       0\n",
       "Marginal Adhension             0\n",
       "Single Epithelial Cell Size    0\n",
       "Bare Nuclei                    0\n",
       "Bland Chromatin                0\n",
       "Normal Nucleoli                0\n",
       "Mitoses                        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Checking for missing values in the training set after replacing with the median value.\n",
    "x_train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "f86b7463-fd5d-4d89-81b6-26070bbc36c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clump Thickness                0\n",
       "Uniformity of Cell Size        0\n",
       "Uniformity of Cell Shape       0\n",
       "Marginal Adhension             0\n",
       "Single Epithelial Cell Size    0\n",
       "Bare Nuclei                    0\n",
       "Bland Chromatin                0\n",
       "Normal Nucleoli                0\n",
       "Mitoses                        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Checking for missing values in the test set after replacing with the median value.\n",
    "x_test.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94523071-5ebb-4b86-b1a3-eed614f672f3",
   "metadata": {},
   "source": [
    "# Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "9c68c3d4-a5d7-4e93-bff3-8ea1f91c8975",
   "metadata": {},
   "outputs": [],
   "source": [
    "cols= x_train.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "02d728ef-b5e6-41f2-9138-9f0fbca8f0d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler=StandardScaler()\n",
    "x_train= scaler.fit_transform(x_train)\n",
    "x_test = scaler.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "18a2cdca-77ab-421f-bd0f-22aa2ea0a228",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = pd.DataFrame(x_train, columns=[cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "73f144bf-0639-4f5e-8d51-5283b7f4bd33",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_test = pd.DataFrame(x_test, columns=[cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c1159261-d722-461f-b1e6-2615bfefa62e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>Clump Thickness</th>\n",
       "      <th>Uniformity of Cell Size</th>\n",
       "      <th>Uniformity of Cell Shape</th>\n",
       "      <th>Marginal Adhension</th>\n",
       "      <th>Single Epithelial Cell Size</th>\n",
       "      <th>Bare Nuclei</th>\n",
       "      <th>Bland Chromatin</th>\n",
       "      <th>Normal Nucleoli</th>\n",
       "      <th>Mitoses</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.028383</td>\n",
       "      <td>0.299506</td>\n",
       "      <td>0.289573</td>\n",
       "      <td>1.119077</td>\n",
       "      <td>-0.546543</td>\n",
       "      <td>1.858357</td>\n",
       "      <td>-0.577774</td>\n",
       "      <td>0.041241</td>\n",
       "      <td>-0.324258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.669451</td>\n",
       "      <td>2.257680</td>\n",
       "      <td>2.304569</td>\n",
       "      <td>-0.622471</td>\n",
       "      <td>3.106879</td>\n",
       "      <td>1.297589</td>\n",
       "      <td>-0.159953</td>\n",
       "      <td>0.041241</td>\n",
       "      <td>-0.324258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.202005</td>\n",
       "      <td>-0.679581</td>\n",
       "      <td>-0.717925</td>\n",
       "      <td>0.074148</td>\n",
       "      <td>-1.003220</td>\n",
       "      <td>-0.104329</td>\n",
       "      <td>-0.995595</td>\n",
       "      <td>-0.608165</td>\n",
       "      <td>-0.324258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.125209</td>\n",
       "      <td>-0.026856</td>\n",
       "      <td>-0.046260</td>\n",
       "      <td>-0.622471</td>\n",
       "      <td>-0.546543</td>\n",
       "      <td>-0.665096</td>\n",
       "      <td>-0.159953</td>\n",
       "      <td>0.041241</td>\n",
       "      <td>-0.324258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.233723</td>\n",
       "      <td>-0.353219</td>\n",
       "      <td>-0.382092</td>\n",
       "      <td>-0.274161</td>\n",
       "      <td>-0.546543</td>\n",
       "      <td>-0.665096</td>\n",
       "      <td>-0.577774</td>\n",
       "      <td>-0.283462</td>\n",
       "      <td>-0.324258</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Clump Thickness Uniformity of Cell Size Uniformity of Cell Shape  \\\n",
       "0        2.028383                0.299506                 0.289573   \n",
       "1        1.669451                2.257680                 2.304569   \n",
       "2       -1.202005               -0.679581                -0.717925   \n",
       "3       -0.125209               -0.026856                -0.046260   \n",
       "4        0.233723               -0.353219                -0.382092   \n",
       "\n",
       "  Marginal Adhension Single Epithelial Cell Size Bare Nuclei Bland Chromatin  \\\n",
       "0           1.119077                   -0.546543    1.858357       -0.577774   \n",
       "1          -0.622471                    3.106879    1.297589       -0.159953   \n",
       "2           0.074148                   -1.003220   -0.104329       -0.995595   \n",
       "3          -0.622471                   -0.546543   -0.665096       -0.159953   \n",
       "4          -0.274161                   -0.546543   -0.665096       -0.577774   \n",
       "\n",
       "  Normal Nucleoli   Mitoses  \n",
       "0        0.041241 -0.324258  \n",
       "1        0.041241 -0.324258  \n",
       "2       -0.608165 -0.324258  \n",
       "3        0.041241 -0.324258  \n",
       "4       -0.283462 -0.324258  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "438f20d6-f921-4e22-b1a3-7dfa9ebafc99",
   "metadata": {},
   "source": [
    "Fitting the Model to the training set\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "54ba4cfe-3591-435e-88b1-e570629831fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importing the libraries\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "#Instatiating the library\n",
    "knn = KNeighborsClassifier(n_neighbors=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "e5df8aff-2acb-4afe-b769-b30e5950dd75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier(n_neighbors=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier(n_neighbors=3)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=3)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1baa58ee-06ea-43a3-ab87-2f2126d14d9d",
   "metadata": {},
   "source": [
    "###Predicting the Test Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "596c7568-8434-48b1-b92d-d0612a1ecc1e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2, 2, 4, 4, 4, 2, 2, 4, 4, 2, 4, 4,\n",
       "       2, 2, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2,\n",
       "       4, 4, 2, 4, 2, 4, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 4, 4,\n",
       "       4, 2, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2,\n",
       "       4, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2,\n",
       "       4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 4, 2, 2, 4, 4, 4, 4, 4, 2,\n",
       "       2, 4, 4, 2, 2, 4, 2, 2], dtype=int64)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predict= knn.predict(x_test)\n",
    "y_predict"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41a69df5-888d-48dc-85ca-13318a2784dc",
   "metadata": {},
   "source": [
    "#Using Predict_Proba Method"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "b32125c5-7f83-444a-b5c3-43535748596b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.        , 1.        , 0.33333333, 1.        , 0.        ,\n",
       "       1.        , 0.        , 1.        , 0.        , 0.66666667,\n",
       "       1.        , 1.        , 0.        , 0.33333333, 0.        ,\n",
       "       1.        , 1.        , 0.        , 0.        , 1.        ,\n",
       "       0.        , 0.        , 1.        , 1.        , 1.        ,\n",
       "       0.        , 1.        , 1.        , 0.        , 0.        ,\n",
       "       1.        , 1.        , 1.        , 1.        , 1.        ,\n",
       "       0.66666667, 1.        , 0.        , 1.        , 1.        ,\n",
       "       1.        , 1.        , 1.        , 1.        , 0.        ,\n",
       "       0.        , 1.        , 0.        , 1.        , 0.        ,\n",
       "       0.        , 1.        , 1.        , 0.        , 1.        ,\n",
       "       1.        , 1.        , 1.        , 0.66666667, 1.        ,\n",
       "       0.        , 1.        , 1.        , 0.        , 0.        ,\n",
       "       0.33333333, 0.        , 1.        , 1.        , 0.        ,\n",
       "       1.        , 1.        , 0.        , 0.        , 1.        ,\n",
       "       1.        , 1.        , 1.        , 0.        , 1.        ,\n",
       "       1.        , 1.        , 0.        , 1.        , 1.        ,\n",
       "       1.        , 0.        , 1.        , 0.        , 0.        ,\n",
       "       1.        , 1.        , 0.66666667, 0.        , 1.        ,\n",
       "       1.        , 1.        , 0.        , 1.        , 0.        ,\n",
       "       0.        , 1.        , 1.        , 1.        , 0.        ,\n",
       "       1.        , 1.        , 1.        , 1.        , 1.        ,\n",
       "       0.        , 0.33333333, 0.        , 1.        , 1.        ,\n",
       "       1.        , 1.        , 1.        , 0.        , 0.        ,\n",
       "       0.        , 0.33333333, 1.        , 0.        , 1.        ,\n",
       "       1.        , 0.33333333, 0.33333333, 0.        , 0.        ,\n",
       "       0.        , 1.        , 1.        , 0.33333333, 0.        ,\n",
       "       1.        , 1.        , 0.        , 1.        , 1.        ])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##predict_proba method is used to find the probability of having benign cancer\n",
    "knn.predict_proba(x_test)[:,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "14e44304-1220-4468-b789-f9f93112002a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 0.        , 0.66666667, 0.        , 1.        ,\n",
       "       0.        , 1.        , 0.        , 1.        , 0.33333333,\n",
       "       0.        , 0.        , 1.        , 0.66666667, 1.        ,\n",
       "       0.        , 0.        , 1.        , 1.        , 0.        ,\n",
       "       1.        , 1.        , 0.        , 0.        , 0.        ,\n",
       "       1.        , 0.        , 0.        , 1.        , 1.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       0.33333333, 0.        , 1.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 1.        ,\n",
       "       1.        , 0.        , 1.        , 0.        , 1.        ,\n",
       "       1.        , 0.        , 0.        , 1.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 0.33333333, 0.        ,\n",
       "       1.        , 0.        , 0.        , 1.        , 1.        ,\n",
       "       0.66666667, 1.        , 0.        , 0.        , 1.        ,\n",
       "       0.        , 0.        , 1.        , 1.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 1.        , 0.        ,\n",
       "       0.        , 0.        , 1.        , 0.        , 0.        ,\n",
       "       0.        , 1.        , 0.        , 1.        , 1.        ,\n",
       "       0.        , 0.        , 0.33333333, 1.        , 0.        ,\n",
       "       0.        , 0.        , 1.        , 0.        , 1.        ,\n",
       "       1.        , 0.        , 0.        , 0.        , 1.        ,\n",
       "       0.        , 0.        , 0.        , 0.        , 0.        ,\n",
       "       1.        , 0.66666667, 1.        , 0.        , 0.        ,\n",
       "       0.        , 0.        , 0.        , 1.        , 1.        ,\n",
       "       1.        , 0.66666667, 0.        , 1.        , 0.        ,\n",
       "       0.        , 0.66666667, 0.66666667, 1.        , 1.        ,\n",
       "       1.        , 0.        , 0.        , 0.66666667, 1.        ,\n",
       "       0.        , 0.        , 1.        , 0.        , 0.        ])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##predict_proba method is used to find the probability of having malignant cancer\n",
    "knn.predict_proba(x_test)[:,1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a63fb0c1-7092-4573-8a06-ec8fcc27aa7d",
   "metadata": {},
   "source": [
    "Checking the Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "1d0336e3-36ae-4cd2-8095-8dbae7f8bf66",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The accuracy score is:0.9714\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "print('The accuracy score is:{0:0.4f}'.format(accuracy_score(y_Test,y_predict)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38655c9a-b8a9-4de4-b61b-fd3e81cfdfb4",
   "metadata": {},
   "source": [
    "#Comparing the accuracy of the test set with the the accuracy of the training set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "0f23e541-39ad-4993-b09a-a2eb870fb94c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Predicting the output of the training set\n",
    "y_pred_training = knn.predict(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "47877133-8c50-4089-a0bc-b7d33fb2a451",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The accuracy score is: 0.9821\n"
     ]
    }
   ],
   "source": [
    "#Checking the accuracy of the prediction from the training set\n",
    "print('The accuracy score is: {0:0.4f}'.format(accuracy_score(y_train,y_pred_training)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05e4f89c-c410-40c1-b43a-80975e390cf3",
   "metadata": {},
   "source": [
    "#Checking for Overfitting and Underfitting"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee2d66a1-5428-4eef-8bf9-e1f9e6ec49bb",
   "metadata": {},
   "source": [
    "Checking for Null accuracy \n",
    "Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.\n",
    "It is calculated by dividing the number of the most frequent class by the total number of output value in the test set.\n",
    "To get this, we'll first count the frequency of each class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "3628c188-d795-413e-80d2-98a2610bcc56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Clas\n",
       "2    85\n",
       "4    55\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Finding the frequency of each class\n",
    "y_Test.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "b1c6f99f-4a90-4c97-af30-e217f8f947cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Null Accuracy of the test data is:0.6071\n"
     ]
    }
   ],
   "source": [
    "#Calculating the null accuracy\n",
    "Null_Accuracy= 85/(85+55)\n",
    "print('Null Accuracy of the test data is:{0:0.4f}'.format(Null_Accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d742102e-8d80-49d1-b067-4873c3db70ee",
   "metadata": {},
   "source": [
    "Since the null accuracy is lower than the accuracy of the predicted value, \n",
    "then we can conlude our model is doing a very good job. \n",
    "However we will rebuild the model with a higher k value to see which gives a better accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "8659fcb4-dbaf-492a-a0a8-5642eda13d2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Rebuilding KNN where k=5\n",
    "knn5=KNeighborsClassifier(n_neighbors=5)\n",
    "knn5.fit(x_train, y_train)\n",
    "y_pred2= knn5.predict(x_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "2843065c-4ea4-43e9-887a-d8efe1927848",
   "metadata": {},
   "outputs": [],
   "source": [
    "#finding the accuracy score for the model\n",
    "Acc= accuracy_score(y_pred2, y_Test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "d7e5a2db-38ec-4d97-a561-870e95199a3f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy score of k=5 is 0.9714\n"
     ]
    }
   ],
   "source": [
    "print('The Accuracy score of k=5 is {0:0.4f}'.format(Acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "3376eab2-3dc2-446f-a83e-f84dbacbb937",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy Score for K=6 is 0.9786\n"
     ]
    }
   ],
   "source": [
    "#Rebuilding KNN where K=6\n",
    "knn6=KNeighborsClassifier(n_neighbors=6)\n",
    "knn6.fit(x_train, y_train)\n",
    "y_pred6= knn6.predict(x_test)\n",
    "\n",
    "#Checking the Accuracy Score for the model where K=6\n",
    "print('The Accuracy Score for K=6 is {0:0.4f}'.format(accuracy_score(y_pred6, y_Test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "713d05be-4216-4792-885e-3e0976555884",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy Score for K=7 is 0.9786\n"
     ]
    }
   ],
   "source": [
    "#Rebuilding KNN where K=7\n",
    "knn7=KNeighborsClassifier(n_neighbors=7)\n",
    "knn7.fit(x_train, y_train)\n",
    "y_pred7=knn7.predict(x_test)\n",
    "\n",
    "#Checking the Accuracy Score for the model where K=7\n",
    "print('The Accuracy Score for K=7 is {0:0.4f}'.format(accuracy_score(y_pred7,y_Test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "557f42f0-dd58-49b6-a82e-232c5cc299c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy Score for k=8 is 0.9786\n"
     ]
    }
   ],
   "source": [
    "#Rebuilding KNN where K=8\n",
    "knn8=KNeighborsClassifier(n_neighbors=8)\n",
    "knn8.fit(x_train, y_train)\n",
    "y_pred8=knn8.predict(x_test)\n",
    "\n",
    "#Checking the Accuracy Score for the model where K=8\n",
    "print('The Accuracy Score for k=8 is {0:0.4f}'.format(accuracy_score(y_pred8, y_Test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "ba995040-0fc1-414e-9378-4ac4a35d62f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy Score for k=9 is 0.9714\n"
     ]
    }
   ],
   "source": [
    "#Rebuilding KNN where K=9\n",
    "knn9=KNeighborsClassifier(n_neighbors=9)\n",
    "knn9.fit(x_train, y_train)\n",
    "y_pred9=knn9.predict(x_test)\n",
    "\n",
    "#Checking the Accuracy Score for the model where K=9\n",
    "print('The Accuracy Score for k=9 is {0:0.4f}'.format(accuracy_score(y_pred9, y_Test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "03260f68-6550-4533-aeae-dcb578d624d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion matrix \n",
      "\n",
      " [[83  2]\n",
      " [ 2 53]]\n",
      "True Negative\n",
      "= 53\n",
      "False Negative\n",
      "= 2\n",
      "False Positive\n",
      "= 2\n",
      "True Positive\n",
      "=  83\n"
     ]
    }
   ],
   "source": [
    "#Computing the Confusion Matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "#Fitting the confusion matrix for K=3\n",
    "cm=confusion_matrix(y_Test, y_predict)\n",
    "\n",
    "#print the output\n",
    "print('confusion matrix \\n\\n', cm)\n",
    "print('True Negative\\n=',cm[1,1])\n",
    "print('False Negative\\n=', cm[1,0])\n",
    "print('False Positive\\n=', cm[0,1])\n",
    "print('True Positive\\n= ', cm[0,0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "b65a3483-717a-4df4-bc9e-e3186d8fbadd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix is\n",
      "\n",
      " [[83  1]\n",
      " [ 2 54]]\n",
      "True Positive is:\n",
      " 83\n",
      "True Negative is:\n",
      " 54\n",
      "False Positve is:\n",
      " 1\n",
      "False Negative is: \n",
      " 2\n"
     ]
    }
   ],
   "source": [
    "#Computing the confusion matrix for K=7\n",
    "cm7=confusion_matrix(y_pred7, y_Test)\n",
    "\n",
    "print('Confusion Matrix is\\n\\n', cm7)\n",
    "print('True Positive is:\\n', cm7[0,0])\n",
    "print('True Negative is:\\n', cm7[1,1])\n",
    "print('False Positve is:\\n', cm7[0,1])\n",
    "print('False Negative is: \\n', cm7[1,0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8343c50a-b8b2-4fde-afd9-c480ae7f1254",
   "metadata": {},
   "source": [
    "###Visualising the Confusion Matrix of the updated data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "7f291a36-27af-4f16-acac-a102275876b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(6,4))\n",
    "CM_Plt=pd.DataFrame(data=cm7, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predicted positive:1', 'Predicted negatives:0'])\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')                                                                                 "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0327ed26-3655-4160-a8fb-e706567f0e0e",
   "metadata": {},
   "source": [
    "#Evaluating the model's performance using classification report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "c8ddb59c-4df8-4709-90b4-8c6d76414443",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           2       0.99      0.98      0.98        85\n",
      "           4       0.96      0.98      0.97        55\n",
      "\n",
      "    accuracy                           0.98       140\n",
      "   macro avg       0.98      0.98      0.98       140\n",
      "weighted avg       0.98      0.98      0.98       140\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_Test, y_pred7))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe1a3884-0ab6-4c9d-9509-5e777f789ace",
   "metadata": {},
   "source": [
    "#Calculating the classification Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "1740bb7d-cdac-4043-8aef-485e90ae9850",
   "metadata": {},
   "outputs": [],
   "source": [
    "TP= cm7[0,0]\n",
    "TN= cm7[1,1]\n",
    "FP= cm7[0,1]\n",
    "FN= cm7[1,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "a72b4f0e-f5b6-4530-9535-170baceebf63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The model classification accuracy is 0.9786\n"
     ]
    }
   ],
   "source": [
    "#Classification Accuracy\n",
    "classification_accuracy= (TP+TN)/float(TP+TN+FP+FN)\n",
    "print('The model classification accuracy is {0:0.4f}'.format(classification_accuracy))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "13e90f0f-d546-43a6-9004-e95079c990bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The model classification error is 0.0214\n"
     ]
    }
   ],
   "source": [
    "#Classification Error\n",
    "classification_error= (FP+FN)/float(TP+TN+FP+FN)\n",
    "print('The model classification error is {0:0.4f}'.format(classification_error))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "995624e8-f4df-45b0-b142-ec55c051d8b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Precision of model is 0.9881\n"
     ]
    }
   ],
   "source": [
    "#Calculating the Precision\n",
    "precision= TP/float(TP+FP)\n",
    "print('The Precision of model is {0:0.4f}'.format(precision))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "24b816d7-753a-48ca-b630-70a1e459f58f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recall is 0.9765\n"
     ]
    }
   ],
   "source": [
    "#Calculating the Recall\n",
    "Recall = TP/float(TP+FN)\n",
    "print('Recall is {0:0.4f}'.format(Recall))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "18ccaec1-42a9-4aac-a113-b26ca557bec6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False Positive Rate is 0.0182\n"
     ]
    }
   ],
   "source": [
    "#Calculating False Positive Rate\n",
    "False_Positive_Rate= FP/float(FP+TN)\n",
    "print('False Positive Rate is {0:0.4f}'.format(False_Positive_Rate))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "edd08dfa-8db6-4c3e-a528-9a165e953ea9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Specificity : 0.9818\n"
     ]
    }
   ],
   "source": [
    "specificity = TN / (TN + FP)\n",
    "\n",
    "print('Specificity : {0:0.4f}'.format(specificity))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "bc23ed45-d103-4523-8efd-32b30fcbdcd4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.        , 0.        ],\n",
       "       [1.        , 0.        ],\n",
       "       [0.33333333, 0.66666667],\n",
       "       [1.        , 0.        ],\n",
       "       [0.        , 1.        ],\n",
       "       [1.        , 0.        ],\n",
       "       [0.        , 1.        ],\n",
       "       [1.        , 0.        ],\n",
       "       [0.        , 1.        ],\n",
       "       [0.66666667, 0.33333333]])"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print the first 10 predicted probabilities of two classes- 2 and 4\n",
    "\n",
    "y_pred_prob = knn.predict_proba(x_test)[0:10]\n",
    "\n",
    "y_pred_prob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a84cb120-f89b-43f0-a32f-aaae139cf855",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
n.ipynb)

