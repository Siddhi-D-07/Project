# Project : Predicting Fuel Efficiency of Indian Vehicles Using Deep Learning ##

# Goal:
To create and add meaningful new features (like power-to-weight ratio or average annual kilometers) that enhance the dataset, using domain knowledge and later evaluating their effect on model performance.

# Feature Engineering involves:

- Creating new features from existing ones (e.g., power-to-weight ratio).

- Using domain knowledge to ensure these features make sense and are meaningful.

- Evaluating their impact by checking if they improve the model's accuracy or performance.

# Steps performed for Feature Engineering

## 1.Derive additional features.

*Creating new columns from existing ones to help machine learning model perform better.*

<pre>df = dataset.copy()</pre>

- This line creates a copy of the original dataset (dataset) and stores it in df, so that any changes made to df won’t affect the original data.

#### Feature 1:Power to weight ratio
#### Combines engine strength and vehicle weight


<pre>df['power_to_weight']=df['horsepower']/df['vehicle_weight']</pre>

- This line creates a new feature `power_to_weight` by dividing a vehicle's horsepower by its weight, combining engine power and weight to show how powerful the vehicle is relative to its size.

#### Feature 2:hp per liter
#### Horsepower per liter (engine efficiency indicator)


<pre>df["hp_per_liter"] = df["horsepower"] / df["engine_size"]</pre>

- This line creates the `hp_per_liter` feature by dividing horsepower by engine size, which indicates how efficiently the engine produces power per liter.

#### Features 3:Engine capacity ratio
#### Engine capacity vs weight  (bigger engine in a light car may waste fuel)
<pre>df["engine_capacity_ratio"]= df["engine_capacities"] / df["vehicle_weight"]</pre>

- This line creates the engine_capacity_ratio feature by dividing engine capacity by vehicle weight, showing how large the engine is relative to the car’s weight—which can affect fuel efficiency.

#### Feature 4: Vehicle age
#### Older vehicles may have lower efficiency
<pre> current_year = 2025
df["vehicle_age"] = current_year - df["model_year"]</pre>

- This line creates the `vehicle_age` feature by subtracting the vehicle’s model year from the current year (2025), indicating how old the vehicle is, which may affect its efficiency and performance.


<pre>df[["power_to_weight", "hp_per_liter","engine_capacity_ratio", "vehicle_age"]].head()</pre>

- This line displays the first few rows of the newly created features—`power_to_weight`, `hp_per_liter`, `engine_capacity_ratio`, and `vehicle_age`—to verify that they were added correctly to the dataset.

*Output*

<pre> 
	power_to_weight	       hp_per_liter	engine_capacity_ratio	    vehicle_age
0	0.241786	        421.916667	        0.000478	            31
1	0.083652	        78.184358	        0.000897	            32
2	0.105991	        77.567568	        0.001641	            27
3	0.502642	        123.219048	        0.000777	            16
4	0.234734	        361.278195	        0.002443	            35 
</pre>

## 2.Evaluate the impact of your engineered features

<pre> correlation = df["power_to_weight"].corr(dataset["fuel_efficiency"])
print("Correlation between  power_to_weight  and Fuel Efficiency:", correlation) </pre>

- This code checks the linear relationship between `power_to_weight` and `fuel_efficiency` using the `.corr()` function to see if the new feature is relevant for predicting fuel efficiency.

*Output: Correlation between  power_to_weight  and Fuel Efficiency: -0.0005065398000716088

<pre>print("Correlation between  hp_per_liter  and Fuel Efficiency:", correlation)</pre>

- This code checks how strongly `hp_per_liter` is related to `fuel_efficiency` using correlation, helping determine if the feature is useful for prediction.

*Output:Correlation between  hp_per_liter  and Fuel Efficiency: 0.0005295783299152598*

<pre>correlation = df["engine_capacity_ratio"].corr(dataset["fuel_efficiency"])
print("Correlation between  engine_capacity_ratio  and Fuel Efficiency:", correlation)</pre>

- This code calculates the correlation between `engine_capacity_ratio` and `fuel_efficiency` to check if there's a meaningful relationship. It helps assess whether this new feature can be useful in predicting fuel efficiency.

*Output:Correlation between  engine_capacity_ratio  and Fuel Efficiency: -0.0011430971022186865*

<pre>correlation = df["vehicle_age"].corr(dataset["fuel_efficiency"])
print("Correlation between  vehicle_age  and Fuel Efficiency:", correlation)</pre>

- This code measures the correlation between `vehicle_age` and `fuel_efficiency` to see how vehicle age affects fuel efficiency. It helps evaluate if older vehicles tend to be less efficient and whether this feature should be used in the model.

*Output:Correlation between  vehicle_age  and Fuel Efficiency: -0.0004983299551720108*

## Final Integration Step

- Adding all the newly created features (power_to_weight, hp_per_liter, engine_capacity_ratio, vehicle_age) into your main dataset (dataset) so that they can be used for training and evaluating for further ML model.

<pre>dataset = dataset.join(df[['power_to_weight', 'hp_per_liter', 'engine_capacity_ratio', 'vehicle_age']])</pre>

- This line adds the new engineered features (`power_to_weight`, `hp_per_liter`, `engine_capacity_ratio`, and `vehicle_age`) from `df` into the main `dataset`, preparing it for model training.

<pre>dataset.info()</pre>

*Output:*
<pre>  <class 'pandas.core.frame.DataFrame'>
RangeIndex: 401000 entries, 0 to 400999
Data columns (total 16 columns):
 #   Column                 Non-Null Count   Dtype  
---  ------                 --------------   -----  
 0   vehicle_id             401000 non-null  int64  
 1   engine_size            401000 non-null  float64
 2   horsepower             401000 non-null  float64
 3   vehicle_weight         401000 non-null  int64  
 4   model_year             401000 non-null  int64  
 5   fuel_type              401000 non-null  object 
 6   vehicle_type           401000 non-null  object 
 7   fuel_efficiency        401000 non-null  float64
 8   engine_capacities      401000 non-null  float64
 9   transmission_types     401000 non-null  object 
 10  road_conditions        401000 non-null  object 
 11  climate_factors        401000 non-null  object 
 12  power_to_weight        401000 non-null  float64
 13  hp_per_liter           401000 non-null  float64
 14  engine_capacity_ratio  401000 non-null  float64
 15  vehicle_age            401000 non-null  int64  
dtypes: float64(7), int64(4), object(5)
memory usage: 49.0+ MB</pre>

# Summary:

To enhance the dataset and improve model performance, we engineered four new features using domain knowledge: power_to_weight, hp_per_liter, engine_capacity_ratio, and vehicle_age. Each was derived from existing columns to reflect factors affecting fuel efficiency, such as engine strength, vehicle weight, engine size, and age. We evaluated their relevance using correlation with the target (fuel_efficiency) and found weak linear relationships, but these features may still add value when combined with others in a machine learning model. Finally, we integrated these new features into the main dataset for further analysis and model training.