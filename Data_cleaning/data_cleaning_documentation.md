# Project: Predicting Fuel Efficiency of Indian Vehicles Using Deep Learning ##

# Goal
To inspect the dataset, check for missing values, and remove any incomplete rows to ensure the data is clean, complete, and ready for analysis.

# How we started cleaning the data:
To begin the data cleaning process, we first explored the dataset to identify potential issues. This included checking for missing values, null entries, spelling mistakes, and any inconsistencies in the data.

# Steps used to clean the data:
## 1.Identify and handle missing values, outliers, and inconsistencies:

### Missing Values:

- Ensure no important information is lost or skipped. Missing data can cause errors or reduce model accuracy.

<pre>dataset.isnull().sum()</pre>

- This command checked missing (null) values in each column of the dataset and shows how many are missing.

<pre>dataset.dropna()</pre> 

- dataset.dropna() removes rows with missing values to ensure clean, complete data for analysis. Use it when missing data is minimal to avoid losing important information.

<pre>dataset.isnull().sum()</pre>

- We used dataset.isnull().sum() again to confirm that no missing values remained in the dataset.

### Inconsistencant values:

Fix spelling, formatting, or category errors to keep data uniform and reliable for analysis.

<pre> dataset[fuel_type] = dataset['fuel_type.replace]({
    'diesel' : 'Diesel',
    'petrol' : 'Petrol'
     'lpg'   : 'LPG'
})</pre>


## 2.Validate plausible ranges and values.

Validated plausible ranges to ensure that all values are realistic and within expected limits, helping maintain data accuracy.

- Check if values are within valid range(horsepower)

<pre>print('Horsepower range:', dataset['horsepower].min(), 'to', dataset['horsepower'].max())</pre>

*Output:  Horsepower range: 60.0 to 700.0*

- Check if values are within valid range(engine size)

<pre>print('engine size:', dataset['engine_size'].min(), 'to', dataset['engine_size'].max())</pre>

*Output:   Engine size range: 1.0 to 6.0*

- Check if values are within valid range(fuel efficiency)

<pre>print('engine size:', dataset['fuel_efficiency'].min(), 'to', dataset['fuel_efficiency'].max())</pre>

*Output:  Fuel size range: 5.0 to 40.0*

- Check if values are within valid range(engine capacities)

<pre>print('engine size:', dataset['engine_capacities'].min(), 'to', dataset['engine_capacities'].max())</pre>

*Output:  Engine capacities range: 0.0 to 6.0*

## Visual Check with Boxplot

<pre>import matplotlib.pyplot as plt

#List of numeric columns you want to check
columns = ['horsepower', 'fuel_efficiency', 'engine_size','engine_capacities']

#Loop through each and plot
for col in columns:
    plt.figure(figsize=(8, 5))
    plt.hist(dataset[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()</pre>

This above code uses Matplotlib to create histograms for selected numeric columns (horsepower, fuel_efficiency, engine_size, and engine_capacities) from the dataset. It helps visualize the distribution and frequency of values in each column to detect patterns or outliers.

*Output*

<p align="center">
  <img src="../Screenshots/horsepower.png" alt="Distribution of Fuel Efficiency" width="700">
</p>

<p align="center">
  <img src="../Screenshots/fuel.png" alt="Distribution of Fuel Efficiency" width="700">
</p>

<p align="center">
  <img src="../Screenshots/engine size.png" alt="Distribution of Fuel Efficiency" width="700">
</p>

<p align="center">
  <img src="../Screenshots/engine capacity.png" alt="Distribution of Fuel Efficiency" width="700">

*Output:
count    12025.000000
mean         3.272121
std          1.600635
min          0.000000
25%          1.940000
50%          3.240000
75%          4.660000
max          6.000000
Name: engine_capacities, dtype: float64*

# Summary 

This data cleaning project aimed to prepare a large dataset of Indian vehicles for deep learning analysis. The main steps included identifying and handling missing values using dropna(), correcting inconsistent values (like fuel type spelling errors), and validating numeric columns to ensure all values were within realistic ranges. Key features such as horsepower, engine size, and fuel efficiency were visually inspected using histograms to detect outliers. Finally, statistical summaries confirmed data distribution, ensuring the dataset is clean, consistent, and ready for modeling.