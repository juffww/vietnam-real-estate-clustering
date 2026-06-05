# Vietnam Real Estate Market Segmentation

This project applies unsupervised machine learning to cluster Vietnam real estate listings into meaningful market segments. The main goal is to group real estate listings with similar characteristics and support market segmentation analysis based on price, area, property type, location, and listing descriptions.

## Dataset

This project uses the **Tinix Vietnam Real Estate Listings** dataset from Hugging Face:

**Source:** `tinixai/vietnam-real-estates`

The dataset contains large-scale real estate listings in Vietnam. It includes tabular, text, and geospatial information about property listings, such as property title, description, property type, province, district, ward, price, area, number of floors, bedrooms, bathrooms, directions, and publication time.

According to the dataset page, the data is provided in Parquet format, uses Vietnamese text, and contains around 3.5 million rows in the training split. The dataset is licensed under CC BY-NC 4.0.

Although the original dataset is tagged for tabular regression and price prediction, this project uses it for an unsupervised learning task: clustering real estate listings into meaningful market segments.

Due to the large size of the dataset, the raw data is not included in this repository. Users can download it directly from Hugging Face and place it in a local `data/` folder before running the notebook.

Link dataset: https://huggingface.co/datasets/tinixai/vietnam-real-estates

## Project Objectives

* Analyze Vietnam real estate listing data.
* Clean and preprocess raw real estate data.
* Extract useful features from numerical, categorical, geographical, and text-based attributes.
* Apply clustering algorithms to segment real estate listings.
* Evaluate clustering quality using unsupervised learning metrics.
* Interpret clusters to support real estate market segmentation.

## Main Workflow

### 1. Data Understanding and Exploratory Data Analysis

The dataset contains millions of real estate listings with multiple attributes such as property type, province, district, ward, price, area, number of bedrooms, number of bathrooms, floor count, and listing descriptions.

Exploratory data analysis was performed to understand:

* Missing value distribution
* Property type distribution
* Province-level listing distribution
* Price distribution
* Area distribution
* Price variation by property type
* Train/test distribution comparison

### 2. Data Cleaning

The data cleaning process includes:

* Removing columns with very high missing rates and low contribution to clustering
* Standardizing data types
* Removing invalid records
* Removing duplicate listings
* Removing records with missing critical information
* Removing records with non-positive price or area values
* Cleaning Vietnamese text descriptions by lowercasing text, removing URLs, phone numbers, and redundant spaces

### 3. Missing Value Handling

Some missing values were handled using rule-based methods. For example, missing `floor_count` values were processed based on property type and information extracted from listing descriptions using regular expressions.

### 4. Outlier Filtering

Outliers were filtered using percentile-based thresholds. The 1st and 99th percentiles from the training set were used to define valid ranges, reducing the impact of extreme price and area values.

### 5. Feature Engineering

Several groups of features were created:

#### Numerical Features

* Log-transformed price
* Log-transformed area
* Log-transformed price per square meter
* Bedroom count
* Bathroom count
* Floor count

Log transformation was applied because price and area distributions were highly right-skewed.

#### Categorical Features

* Property type
* Province
* District
* Ward

Categorical variables were encoded using One-Hot Encoding.

#### Binary Text-based Features

Binary features were extracted from listing descriptions using keyword-based regular expressions. Each feature indicates whether the description contains specific real estate-related keywords.

#### Text Features

The cleaned listing description was represented using TF-IDF. The TF-IDF matrix was then reduced using TruncatedSVD to make the text representation more compact and suitable for clustering.

### 6. Preprocessing Pipeline

A preprocessing pipeline was built using Scikit-learn tools such as `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `TfidfVectorizer`, and `TruncatedSVD`.

The pipeline combines:

* Imputation for missing values
* Log transformation for skewed numerical features
* One-Hot Encoding for categorical features
* TF-IDF vectorization for text descriptions
* TruncatedSVD for dimensionality reduction
* StandardScaler for feature scaling

### 7. Clustering Models

The project experiments with unsupervised clustering models, including:

* MiniBatchKMeans
* Agglomerative Clustering

MiniBatchKMeans was used because it is more suitable for large-scale datasets compared to standard KMeans.

### 8. Model Evaluation

Clustering quality was evaluated using common unsupervised learning metrics:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

These metrics help compare clustering results and select a suitable number of clusters.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Project Structure

```text
vietnam-real-estate-clustering/
├── vietnam_real_estate_clustering.ipynb
├── README.md
├── requirements.txt
├── data/
└── images/
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/juffww/vietnam-real-estate-clustering.git
cd vietnam-real-estate-clustering
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from the original source and place it in the `data/` folder.

4. Open the notebook:

```bash
jupyter notebook vietnam_real_estate_clustering.ipynb
```

or open it with Google Colab.

## Results and Insights

The clustering pipeline groups real estate listings into market segments based on a combination of price, area, property type, location, and listing description features. The resulting clusters can help analyze different real estate segments such as affordable properties, high-value properties, apartments, land listings, and location-based market groups
