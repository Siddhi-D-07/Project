# Project : Predicting Fuel Efficiency of Indian Vehicles Using Deep Learning ##

# Goal:
To transform numerical features into a similar range using techniques like normalization or standardization. This helps deep learning models train faster, converge better, and avoid bias toward features with larger values.

# Purpose of Feature Scaling:
The main purpose of feature scaling is to transform numerical features into a common scale without distorting differences in the ranges of values. It ensures that features contribute equally to the model's learning process, improves training stability, and enhances the performance of algorithms like deep learning and gradient descent-based models.

## 1. Standardizing and Normalizing the Feature Ranges for Model Training

<pre>dataset.head()</pre> 

- It shows the first five rows of the dataset to quickly preview the data and check if it’s loaded correctly.

<pre> from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() </pre>

- This code imports the `LabelEncoder` from `sklearn.preprocessing` and creates an instance named `le`. `LabelEncoder` is used to convert categorical text labels into numeric values, which are easier for machine learning models to process.

- Now , to enable machine learning models to process categorical data, all non-numeric features in the dataset were encoded using LabelEncoder from sklearn.preprocessing. This technique assigns unique integer values to each category in a feature, allowing models to interpret and learn from these variables effectively. The encoded features include fuel_type, vehicle_type, climate_factors, transmission_types, and road_conditions.

<pre>dataset['transmission_types'] = le.fit_transform(dataset['transmission_types'])</pre>

<pre>dataset['road_conditions'] = le.fit_transform(dataset['road_conditions']) </pre>

<pre>dataset['climate_factors'] = le.fit_transform(dataset['climate_factors'])</pre>

<pre>dataset['fuel_type'] = le.fit_transform(dataset['fuel_type'])</pre>

<pre> dataset['vehicle_type'] = le.fit_transform(dataset['vehicle_type'])</pre>

<pre>X = dataset.drop(columns=['vehicle_id', 'fuel_efficiency'], axis=1)
y = dataset['fuel_efficiency']</pre>

- The dataset is divided into input features (X) and the target variable (y). Unnecessary columns like vehicle_id and the target fuel_efficiency are excluded from X, while y stores the fuel_efficiency values to be predicted. This prepares the data for model training.

<pre> from sklearn.preprocessing import MinMaxScaler

minmax_scaler=MinMaxScaler()
minmax=minmax_scaler.fit_transform(X)
minmaxscaler_df=pd.DataFrame(X)

minmaxscaler_df.head() </pre>

- This code applies Min-Max Scaling to normalize all numerical features in the dataset to a range between 0 and 1. This ensures that each feature contributes equally during model training, preventing bias from variables with larger magnitudes.

<pre>from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) </pre>

- This code uses `StandardScaler` to standardize the features in `X` by removing the mean and scaling to unit variance. It transforms the data so that each feature has a mean of 0 and a standard deviation of 1, which is beneficial for many machine learning algorithms, especially those sensitive to feature scales.

## 2.Apply PCA

PCA (Principal Component Analysis) is applied to reduce the dimensionality of a dataset while retaining most of its important information. By transforming correlated features into a smaller set of uncorrelated components, PCA helps improve computational efficiency, reduces overfitting, and enhances model performance by focusing on the most significant variance in the data.

<pre>from sklearn.decomposition import PCA
pca = PCA(n_components=10)</pre>

- This code initializes Principal Component Analysis (PCA) with `n_components=10`, meaning it will reduce the dataset's dimensionality to 10 principal components. PCA helps simplify the dataset by keeping the most important information (variance) while reducing noise and improving model efficiency.

<pre>principal_components = pca.fit_transform(X_scaled)</pre>

- This line applies PCA to the scaled dataset `X_scaled` and transforms it into 10 principal components, storing the result in `principal_components`. This reduces the feature space while preserving the most significant variance in the data, making it more efficient for modeling.

<pre>print(pca.explained_variance_ratio_)</pre>

- This line prints the **explained variance ratio** of each of the 10 principal components derived using PCA. It shows how much of the total variance in the original dataset is captured by each component, helping to understand the importance of each in representing the data.

*Output:[0.17861248 0.14418627 0.13474337 0.09099278 0.07340962 0.07215049
 0.0717072  0.07068156 0.07010454 0.06863263]*

<pre>pca_df=pd.DataFrame(principal_components)</pre>

 - This line converts the PCA output into a pandas DataFrame for easier analysis and further use.

# Summary:

In this project on predicting fuel efficiency of Indian vehicles using deep learning, the dataset is first preprocessed by encoding categorical features with `LabelEncoder`, ensuring that machine learning models can process them effectively. Then, feature scaling is applied using both Min-Max Scaling and Standardization to bring all features to a similar range, which helps the deep learning model train efficiently. After scaling, ` Principal Component Analysis (PCA) ` is used to reduce the dataset’s dimensionality to 10 principal components, preserving most of the variance while simplifying the data. This enhances model performance, speeds up training, and reduces the risk of overfitting.
