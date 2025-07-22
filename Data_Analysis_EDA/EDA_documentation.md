# Project : Predicting Fuel Efficiency of Indian Vehicles Using Deep Learning ##

# Goal:
To create new meaningful features from existing data that can help improve the model’s ability to understand patterns and make accurate predictions.

# Data Analysis :
Data analysis means examining and organizing data to find useful information, patterns, and trends. It helps us understand how different factors (like weight or horsepower) affect outcomes like fuel efficiency. This is done using graphs, statistics, and simple calculations.


# Steps used to Analyse the data:

## 1. Explore Feature Relationships and  Concise interpretation :
- Understand how different columns (features) in your dataset relate to each other, especially the target variable (e.g., fuel efficiency).

### python code -

- Set Seaborn style to white grid background for plots

<pre>import seaborn as sns

sns.set(style='whitegrid')</pre>

- use a scatter plot to quickly see the relationship between engine size and fuel efficiency, helping us understand how one affects the other during EDA.


<pre>sns.scatterplot(data=dataset,x='engine_size',y='fuel_efficiency')
plt.title("Engine_size vs Fuel_efficiency")
plt.xlabel("engine_size")
plt.ylabel("fuel_efficiency")
plt.show()</pre>

*Output*

<p align="center">
  <img src="ScreenShots/Engine_size vs Fuel_efficiency.jpg" alt="Distribution of Horse Power" width="700">
</p>




- use a scatter plot to quickly see the relationship between horsepower and fuel efficiency, helping us understand how increasing engine power might impact fuel consumption during EDA.

<pre>sns.scatterplot(data=dataset,x="horsepower",y="fuel_efficiency")
plt.title("Horsepower vs Fuel_efficiency")
plt.xlabel("Hoursepower")
plt.ylabel("fuel_efficiency")
plt.show()</pre>

*Output*

<p align="center">
  <img src="ScreenShots/Horsepower vs Fuel_efficiency.jpg" alt="Distribution of Horse Power" width="700">
</p>




- use a scatter plot to quickly see the relationship between vehicle weight and fuel efficiency, helping us understand whether heavier vehicles tend to consume more fuel during EDA.

<pre>sns.scatterplot(data=dataset,x="vehicle_weight",y="fuel_efficiency")
plt.title("Vehicle_Weight vs Fuel_Efficiency")
plt.xlabel("Vehicle_Weight")
plt.ylabel("Fuel_Efficiency")
plt.show()</pre>

*Output*

<p align="center">
  <img src="ScreenShots/Vehicle_Weight vs Fuel_Efficiency.jpg" alt="Distribution of Horse Power" width="700">
</p>



- use a scatter plot to explore the relationship between model year and fuel efficiency, helping us understand whether newer cars tend to be more fuel-efficient during EDA.

<pte>sns.scatterplot(data=dataset,x="model_year",y="fuel_efficiency")
plt.title("Model_Year vs Fuel_Efficiency")
plt.xlabel("Model_Year")
plt.ylabel("Fuel_Efficiency")
plt.show()</pre>

*Output*

<p align="center">
  <img src="ScreenShots/Model_Year vs Fuel_Efficiency.jpg" alt="Distribution of Horse Power" width="700">
</p>



- use a scatter plot to analyze the relationship between fuel type and fuel efficiency, helping us see which types of fuel (e.g., petrol, diesel, electric) are associated with better or worse MPG during EDA.

<pre>sns.scatterplot(data=dataset,x="fuel_type",y="fuel_efficiency")
plt.title("Fuel_type vs Fuel_Efficiency")
plt.xlabel("Fuel_type")
plt.ylabel("Fuel_Efficiency")
plt.show()</pre>

*Output*

<p align="center">
  <img src="ScreenShots/Fuel_type vs Fuel_Efficiency.jpg" alt="Distribution of Horse Power" width="700">
</p>


## 2. Hypothesize Possible Explanations Based on Feature Relationships :
- making smart and logical guesses about why certain patterns or trends appear in your data. For example, if you see that older cars have lower fuel efficiency, you might guess it's because of engine wear or outdated technology.

### python code -

- Check how Engine_Size is related to Fuel_Efficiency

<pre>correlation = dataset["engine_size"].corr(dataset["fuel_efficiency"])
print("Correlation between Engine Size and Fuel Efficiency:", correlation)</pre>

*Output*
Correlation between Engine Size and Fuel Efficiency: -0.0022254043115841784

- Check how vehicle_id is related to Fuel_Efficiency

<pre>correlation = dataset["vehicle_id"].corr(dataset["fuel_efficiency"])
print("Correlation between  vehicle_id  and Fuel Efficiency:", correlation)</pre>

*Output*
Correlation between  vehicle_id  and Fuel Efficiency: 0.0005571426362321687

- Check how horsepower is related to Fuel_Efficiency

<pre>correlation = dataset["horsepower"].corr(dataset["fuel_efficiency"])
print("Correlation between  horsepower  and Fuel Efficiency:", correlation)</pre>

*Output*
Correlation between  horsepower  and Fuel Efficiency: -0.00028087745434138457

- Check how vehicle_weight is related to Fuel_Efficiency

<pre>correlation = dataset["vehicle_weight"].corr(dataset["fuel_efficiency"])
print("Correlation between  vehicle_weight  and Fuel Efficiency:", correlation)</pre>

*Output*
Correlation between  vehicle_weight  and Fuel Efficiency: -0.0004508061765527194

- Check how model_year is related to Fuel_Efficiency

<pre>correlation = dataset["model_year"].corr(dataset["fuel_efficiency"])
print("Correlation between  model_year  and Fuel Efficiency:", correlation)</pre>

*Output*
Correlation between  model_year  and Fuel Efficiency: 0.0004983299551720078

- Check how engine_capacities is related to Fuel_Efficiency

<pre>correlation = dataset["engine_capacities"].corr(dataset["fuel_efficiency"])
print("Correlation between  engine_capacities  and Fuel Efficiency:", correlation)</pre>

*Output*
Correlation between  engine_capacities  and Fuel Efficiency: -0.0014159226312785247


# Summary :
We explored feature relationships using visualizations like histograms and scatter plots, followed by calculating correlations to understand how variables like engine size and horsepower relate to fuel efficiency. Based on observed patterns, we hypothesized possible explanations and created new features such as power-to-weight ratio and vehicle age to enhance model performance. Finally, we evaluated these engineered features and found that they helped improve prediction accuracy.


