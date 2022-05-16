# Analysis of Car Crashes throughout Maryland

**Introduction**

The objective of this project is to analyze car crashes throughout Maryland. The counties we will specifically look at throughout Maryland are PG County, Montogomery County, Batimore County, Baltimore City. We will also analyze total crashes in Maryland through the years. 

Throughout the tutorial we plan to analyze and figure out trends with car crashes in Maryland. At the end we hope to figure out if any specific roadscare in need of maintence based on car crashes that occur.

**Libraries Needed**

*   folium
*   requests
*   pandas
*   matplotlib.pyplot
*   LinearRegression
*   PolynomialFeatures 
*   numpy

We highly recommend referring to the following resources for more information about pandas/installation and python 3.6 in general:

1. https://pandas.pydata.org/pandas-docs/stable/install.html
2. https://docs.python.org/3/

# 1. Data collection

We will use the request library to request the opendata API for the JSON file containing The Maryland Statewide Vehicle Crashes from January 2015 through December 2021. Please keep in mind this data set containt more than 771,145 rows. Therefore requesting such a big dataset will take time. Running this can take anywhere from 2-6 min. 





```python
import folium
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
import numpy as np

response = requests.get("https://opendata.maryland.gov/resource/65du-s3qu.json?$limit=800000") 
main_df = pd.DataFrame(response.json())

```

# 2. Data management/representation

We will first set a variable df to equal the main_df. That way if we every need the main dataframe again we can just set a variable to equal main_df instead of get requesting the opendata API.

Next we want to clean up the dataset by removing columns that are unnecessary to our analysis. We will be removing the following columns using the pandas drop() method.


*   county_no 
*   collision_type_code 
*   fix_obj_code
*   light_code
*   muni_code
*   junction_code
*   collision_type_code 
*   surf_cond_code
*   lane_code
*   rd_cond_code
*   rd_div_code
*   fix_obj_code
*   report_no
*   weather_code
*   loc_code
*   area_code
*   harm_event_code1
*   harm_event_code2
*   reference_no
*   reference_type_code
*   reference_suffix
*   reference_road_name
*   feet_miles_flag_desc
*   feet_miles_flag
*   distance_dir_flag
*   rte_no
*   route_type_code
*   logmile_dir_flag_desc
*   logmile_dir_flag
*   rte_suffix
*   :@computed_region_r4de_cuuv
*   junction_desc
*   log_mile
*   c_m_zone_flag
*   signal_flag_desc
*   agency_code
*   signal_flag




```python
df = main_df
df = df.astype({'year':'int'})
df.drop(['county_no', 
         'collision_type_code', 
         'fix_obj_code', 'light_code', 
         'muni_code', 'junction_code',
         'collision_type_code', 'surf_cond_code', 
         'lane_code', 'rd_cond_code', 
         'rd_div_code', 'fix_obj_code', 
         'report_no', 'weather_code', 'loc_code',
         'area_code', 'harm_event_code1', 
         'harm_event_code2', 'reference_no', 'reference_type_code',
         'reference_suffix', 'reference_road_name', 'distance',
         'feet_miles_flag_desc',	'feet_miles_flag',
         'distance_dir_flag',	'rte_no', 'route_type_code',
         'logmile_dir_flag_desc',	'logmile_dir_flag',	'rte_suffix',
         ':@computed_region_r4de_cuuv', 'junction_desc', 'log_mile',
         'c_m_zone_flag', 'signal_flag_desc',
         'agency_code','signal_flag' ], inplace=True, axis=1)

df
```





  <div id="df-7f9bba3c-e261-412d-9a7e-ca8aef6a39c2">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>quarter</th>
      <th>light_desc</th>
      <th>county_desc</th>
      <th>collision_type_desc</th>
      <th>fix_obj_desc</th>
      <th>report_type</th>
      <th>weather_desc</th>
      <th>acc_date</th>
      <th>acc_time</th>
      <th>harm_event_desc1</th>
      <th>harm_event_desc2</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>geocoded_column</th>
      <th>surf_cond_desc</th>
      <th>lane_desc</th>
      <th>rd_cond_desc</th>
      <th>rd_div_desc</th>
      <th>mainroad_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>Q2</td>
      <td>Daylight</td>
      <td>Baltimore</td>
      <td>Other</td>
      <td>Not Applicable</td>
      <td>Property Damage Crash</td>
      <td>Not Applicable</td>
      <td>20200618</td>
      <td>15:15:00</td>
      <td>Parked Vehicle</td>
      <td>Not Applicable</td>
      <td>39.27726285</td>
      <td>-76.5036932</td>
      <td>{'type': 'Point', 'coordinates': [-76.5036932,...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>Q2</td>
      <td>NaN</td>
      <td>Baltimore City</td>
      <td>Other</td>
      <td>Other Pole</td>
      <td>Injury Crash</td>
      <td>NaN</td>
      <td>20200430</td>
      <td>06:39:00</td>
      <td>Other Vehicle</td>
      <td>Other Vehicle</td>
      <td>39.3110247944307</td>
      <td>-76.6164294532046</td>
      <td>{'type': 'Point', 'coordinates': [-76.61642945...</td>
      <td>Dry</td>
      <td>Left Turn Lane</td>
      <td>No Defects</td>
      <td>One-way Trafficway</td>
      <td>CHARLES STREET</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020</td>
      <td>Q2</td>
      <td>Daylight</td>
      <td>Montgomery</td>
      <td>Other</td>
      <td>Not Applicable</td>
      <td>Injury Crash</td>
      <td>NaN</td>
      <td>20200504</td>
      <td>09:46:00</td>
      <td>Pedestrian</td>
      <td>Not Applicable</td>
      <td>39.140680249069</td>
      <td>-77.1934127295612</td>
      <td>{'type': 'Point', 'coordinates': [-77.19341272...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>Q2</td>
      <td>Daylight</td>
      <td>Baltimore City</td>
      <td>Single Vehicle</td>
      <td>Not Applicable</td>
      <td>Injury Crash</td>
      <td>Other</td>
      <td>20170507</td>
      <td>10:39:00</td>
      <td>NaN</td>
      <td>Not Applicable</td>
      <td>39.2829284750108</td>
      <td>-76.6352150952347</td>
      <td>{'type': 'Point', 'coordinates': [-76.63521509...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020</td>
      <td>Q2</td>
      <td>Daylight</td>
      <td>Cecil</td>
      <td>Same Direction Rear End</td>
      <td>Not Applicable</td>
      <td>Property Damage Crash</td>
      <td>NaN</td>
      <td>20200414</td>
      <td>17:32:00</td>
      <td>Other Vehicle</td>
      <td>Not Applicable</td>
      <td>39.6110278333333</td>
      <td>-75.951314</td>
      <td>{'type': 'Point', 'coordinates': [-75.951314, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>771140</th>
      <td>2021</td>
      <td>Q4</td>
      <td>NaN</td>
      <td>Prince George's</td>
      <td>Same Direction Sideswipe</td>
      <td>Not Applicable</td>
      <td>Injury Crash</td>
      <td>NaN</td>
      <td>20211207</td>
      <td>06:10:00</td>
      <td>Other Vehicle</td>
      <td>Not Applicable</td>
      <td>38.9171835586245</td>
      <td>-76.925308968694</td>
      <td>{'type': 'Point', 'coordinates': [-76.92530896...</td>
      <td>Dry</td>
      <td>Right Turn Lane</td>
      <td>No Defects</td>
      <td>Two-way, Divided, Positive Median Barrier</td>
      <td>JOHN HANSON HWY</td>
    </tr>
    <tr>
      <th>771141</th>
      <td>2021</td>
      <td>Q4</td>
      <td>Dark No Lights</td>
      <td>Montgomery</td>
      <td>Same Direction Rear End</td>
      <td>Not Applicable</td>
      <td>Property Damage Crash</td>
      <td>Raining</td>
      <td>20211230</td>
      <td>04:40:00</td>
      <td>Other Vehicle</td>
      <td>Not Applicable</td>
      <td>38.9934997938533</td>
      <td>-77.1577488692777</td>
      <td>{'type': 'Point', 'coordinates': [-77.15774886...</td>
      <td>Wet</td>
      <td>Deceleration Lane</td>
      <td>No Defects</td>
      <td>Two-way, Divided, Positive Median Barrier</td>
      <td>CAPITAL BELTWAY</td>
    </tr>
    <tr>
      <th>771142</th>
      <td>2021</td>
      <td>Q4</td>
      <td>Daylight</td>
      <td>Baltimore</td>
      <td>Same Direction Sideswipe</td>
      <td>Not Applicable</td>
      <td>Property Damage Crash</td>
      <td>Raining</td>
      <td>20211029</td>
      <td>11:40:00</td>
      <td>Other Vehicle</td>
      <td>Not Applicable</td>
      <td>39.3047323327158</td>
      <td>-76.441140820424</td>
      <td>{'type': 'Point', 'coordinates': [-76.44114082...</td>
      <td>Wet</td>
      <td>Left Turn Lane</td>
      <td>No Defects</td>
      <td>NaN</td>
      <td>BACK RIVER NECK RD</td>
    </tr>
    <tr>
      <th>771143</th>
      <td>2021</td>
      <td>Q4</td>
      <td>NaN</td>
      <td>Wicomico</td>
      <td>Same Direction Rear End</td>
      <td>Not Applicable</td>
      <td>Property Damage Crash</td>
      <td>Raining</td>
      <td>20211231</td>
      <td>21:07:00</td>
      <td>Other Vehicle</td>
      <td>Not Applicable</td>
      <td>38.3059147769665</td>
      <td>-75.531586844603</td>
      <td>{'type': 'Point', 'coordinates': [-75.53158684...</td>
      <td>Wet</td>
      <td>Right Turn Lane</td>
      <td>No Defects</td>
      <td>Two-way, Not Divided</td>
      <td>NANTICOKE RD</td>
    </tr>
    <tr>
      <th>771144</th>
      <td>2021</td>
      <td>Q4</td>
      <td>NaN</td>
      <td>Anne Arundel</td>
      <td>Single Vehicle</td>
      <td>Not Applicable</td>
      <td>Injury Crash</td>
      <td>NaN</td>
      <td>20211030</td>
      <td>18:00:00</td>
      <td>Fell Jumped from Motor Vehicle</td>
      <td>Not Applicable</td>
      <td>39.1545099637875</td>
      <td>-76.648470641409</td>
      <td>{'type': 'Point', 'coordinates': [-76.64847064...</td>
      <td>Dry</td>
      <td>Right Turn Lane</td>
      <td>No Defects</td>
      <td>Two-way, Divided, Positive Median Barrier</td>
      <td>NO NAME</td>
    </tr>
  </tbody>
</table>
<p>771145 rows × 20 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7f9bba3c-e261-412d-9a7e-ca8aef6a39c2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7f9bba3c-e261-412d-9a7e-ca8aef6a39c2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7f9bba3c-e261-412d-9a7e-ca8aef6a39c2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# 3. Exploratory data analysis + Hypothesis Testing

In this step we will be analyzing certain types of data. After we analyze we will conduct some hypothesis testing to predict future data points.


First lets take a look and analyze at the number of crashes through the years 2015-2022


```python
data = {}
years = [2015,2016,2017,2018,2019,2020,2021]

for year in years:
    bl = df[df["year"] == year]
    data[year] = bl.size
plt.title("Year vs Number of Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Crashes")
plt.scatter(list(data.keys()), list(data.values()))


```




    <matplotlib.collections.PathCollection at 0x7f8a3dc09f50>




    
![png](output_5_1.png)
    


2016 seems to be the year with the highest amount of car crashes. We can also see that 2020 has the lowest amount of crashes through the years. However, between 2020 and 2021 there seems to be the greatest increase in crashes. This is not good as we want number of crashes to stay low. Another thing we should keep in mind is that 2020 was during Covid-19. Therefore, it makes sense that the number of crashes are much lower as there were less people going outside.

Now lets do some Hypothesis testing on year vs # of crashes by graphing a Linear Regression (i.e. y = ax + b) with the above scatter plot.


```python
lr = LinearRegression()
x = np.array(list(data.keys())).reshape(-1, 1)
y = np.array(list(data.values())).reshape(-1, 1)
lr.fit(x,y)
y_pred = lr.predict(x)
plt.title("Year vs Number of Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Crashes")
plt.scatter(list(data.keys()), list(data.values()))
plt.plot(x, y_pred)
plt.show()
```


    
![png](output_7_0.png)
    


Based on the linear regression line, we predict that that number of crashes in 2022 will be between 2.0 Million to 2.1 Million crashes 


```python
print(lr.score(x,y))
print(lr.predict([[2022]]))
```

    0.3588776313313513
    [[2024277.14285713]]


As you can see, the predicted nunber of crashes in 2022 is 2024277.14285713. This number is between 2.0 Million to 2.1 Million which means our hypothesis is correct. However, the R^2 score of 0.36 does seem a bit low, and it's far from a score of 1. Let's try a Quadratic regression (i.e. y = ax^2 + bx + c).


```python
poly = PolynomialFeatures(degree=2)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Crashes")
plt.scatter(list(data.keys()), list(data.values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_11_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.4769200855039387
    [[1846471.4285202]]


With the Quadratic regression, the number of crashes predicted for 2022 is around 1,846,471 crashes. While the R^2 score of 0.48 is higher than the score for the Linear Regression, the prediction of the number of crashes is lower in the Quadratic Regression compared to the Linear Regression of approximately 2,024,277 crashes. Although this may lead to overfitting, Let's test a Cubic Regression (i.e. y = ax^3 + bx^2 + cx + d) to see if we can get a more realistic prediction.


```python
poly = PolynomialFeatures(degree=3)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Crashes")
plt.scatter(list(data.keys()), list(data.values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_14_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.6199128453013034
    [[2212306.90625]]


With the Cubic Regression, the number of predicted crashes for 2022 is appromiately 2,212,306, which is closer to the Linear Regression's predicted value of 2,024,277 accidents. The R^2 score of 0.62 is much higher than the previous regressions. Although this may lead to future inaccurate predictions due to overfitting, if the predicted value of 2022 crashes in the Cubic Regression is similar to the predicted value of 2022 crashes in the Linear Regression, where we wouldn't risk overfitting if we used that model, and if there is a higher R^2 score that is closer to 1, it is reasonable to say that the Cubic Regression is the best indicator for this data, with a prediction of 2,212,306 crashes in 2022. Also, the overal trend of the data initally seems to be decreasing in number of crashes. However, 2020 was an unpredictable year of low crashes due to Covid-19. Therefore, it would not be safe to say that a Linear Model is appropriate for this dataset if there isn't a somewhat consistent trend. 

We looked at trends of the overall number of crashes, but since the crashes themselves are a bit broad, it is more reasonable to predict the amount of crashes with a specific type. In this dataset, there are three types of crashes: Fatal, Injury, and Propery Damage. We will use the same regressions used above for these types of crashes.

Year vs Number of Fatal Crashes


```python
graph = ['Fatal Crash', 'Injury Crash', 'Property Damage Crash']
lis = []

for ty in graph:
    data = {}
    years = [2016,2017,2018,2019,2020,2021]

    
    for year in years:
        bl = df[df["year"] == year]
        bl = bl[bl['report_type'] == ty]
        data[year] = bl.size

    lis.append(data)


plt.title("Year vs Number of Fatal Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Fatal Crashes")
plt.scatter(list(lis[0].keys()), list(lis[0].values()))
```




    <matplotlib.collections.PathCollection at 0x7f8a3b6f2290>




    
![png](output_19_1.png)
    


Once again, we will start with a Linear Regression.


```python
lr = LinearRegression()
x = np.array(list(lis[0].keys())).reshape(-1, 1)
y = np.array(list(lis[0].values())).reshape(-1, 1)
lr.fit(x,y)
y_pred = lr.predict(x)
plt.title("Year vs Number of Fatal Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Fatal Crashes")
plt.scatter(list(lis[0].keys()), list(lis[0].values()))
plt.plot(x, y_pred)
plt.show()
```


    
![png](output_21_0.png)
    



```python
print(lr.score(x,y))
print(lr.predict([[2022]]))
```

    0.026382898266439025
    [[8681.33333333]]


With the Linear Regression, the predicted number of Fatal Crashes in 2022 are approximately 8,681 crashes with an R^2 score of 0.03, which is incredibly low. The data seems to shift heavily in fatal crashes each year, and oddly, crashes increased in 2020 compared to 2019 during the beginning of the pandemic. Let's see if we can get a better prediction with the Quadratic Regression.


```python
poly = PolynomialFeatures(degree=2)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Fatal Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Fatal Crashes")
plt.scatter(list(lis[0].keys()), list(lis[0].values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_24_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
print(poly_ypred)

```

    0.17202297245627274
    [[8157.99999961]]
    [[8703.57142881]
     [8893.00000015]
     [8970.28571436]
     [8935.4285714 ]
     [8788.42857128]
     [8529.28571403]]


With the Quadratic Regression, the predicted value of Fatal Crashes in 2022 is approximately 8,157 crashes, which is similar to the predicted from the Linear model. The R^2 score is 0.17, which is better from the previous model, but let's see if we can get a more accurate model with the Cubic Regression.


```python
poly = PolynomialFeatures(degree=3)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Fatal Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Fatal Crashes")
plt.scatter(list(lis[0].keys()), list(lis[0].values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_27_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.200147807192667
    [[7683.73400879]]


With the Cubic Regression, the predicted number of Fatal Crashes in 2022 is approximately 7,683 crashes. This is significantly less compared to the previous two predictions, especially since the value is lower than 2021 crashes. The data seems to have the trend of alternating in increasing and decreasing crashes each year. Therefore, this can be an instance of overfitting. Each of these Regressions are not accurate in any case. However, the most accurate seems to be the Linear Regression. The Quadratic and Cubic Regressions seem to decrease the fastest while the Linear Regression is decreasing at a slower rate. This can be better if the data alternates in increasing and decreasing fatal crashes each year, and the R^2 scores are all not close to 1. Therefore, the Linear Regression is the accurate model for this data with a prediction of approximately 8,681 crashes.

Now let's look at the data of Injury Crashes. We will start with the Linear Model.

Year vs Number of Injury Crashes


```python
plt.title("Year vs Number of Injury Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Injury Crashes")
plt.scatter(list(lis[1].keys()), list(lis[1].values()))
```




    <matplotlib.collections.PathCollection at 0x7f8a3b590490>




    
![png](output_32_1.png)
    



```python
lr = LinearRegression()
x = np.array(list(lis[1].keys())).reshape(-1, 1)
y = np.array(list(lis[1].values())).reshape(-1, 1)
lr.fit(x,y)
y_pred = lr.predict(x)
plt.title("Year vs Number of Injury Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Injury Crashes")
plt.scatter(list(lis[1].keys()), list(lis[1].values()))
plt.plot(x, y_pred)
plt.show()
```


    
![png](output_33_0.png)
    



```python
print(lr.score(x,y))
print(lr.predict([[2022]]))
```

    0.7294556195920325
    [[496421.33333333]]


With the Linear Regression, the predicted number of injury crashes in 2022 is approximately 496,421 crashes with an R^2 score of 0.72. This seems to be a good prediction, but let's see what a Quadratic Regression provides.


```python
poly = PolynomialFeatures(degree=2)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Injury Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Injury Crashes")
plt.scatter(list(lis[1].keys()), list(lis[1].values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_36_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.761230444583292
    [[449017.99996948]]


With the Quadratic Regression, the predicted number of accidents in 2022 is approximately 449,017 crashes, with an R^2 score of 0.76. The predictions are getting better, but let's see what a Cubic Regression provides.


```python
poly = PolynomialFeatures(degree=3)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Injury Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Injury Crashes")
plt.scatter(list(lis[1].keys()), list(lis[1].values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_39_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.8468312370307014
    [[608905.0234375]]


With the Cubic Regression, the predicted number of Injury Crashes are approximately 608,905 crashes, with an R^2 score of 0.85. Although the R^2 score is higher, the amount of crashes increase drastically compared to the acual value of 2021 crashes of between 550,000-575,000 crashes. The overall trend of the data seems to be decreasing in crashes, therefore it is safe to say that the Cubic Regression is overfitting in this case. As far as debating the use of the Quadratic Regression vs. the Linear Regression, it would make more sense to use the Linear Regression. Both R^2 scores and predicted values of 2022 injury crashes are similar and closer to 1, therefore it wouldn't make sense to go with a Quadratic regression if there's more of a chance of overfitting the data compared to a Linear Regression. Therefore, the Linear Regression is an accurate model of this dataset with a predicted value of 496,421 injury crashes.

Now let's look at the data of Property Damage Crashes. Let's once again start with the Linear Regression.

Year vs Number of Property Damage Crashes


```python
plt.title("Year vs Number of Property Damage Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Property Damage Crashes")
plt.scatter(list(lis[2].keys()), list(lis[2].values()))
```




    <matplotlib.collections.PathCollection at 0x7f8a3b4b5750>




    
![png](output_44_1.png)
    



```python
lr = LinearRegression()
x = np.array(list(lis[2].keys())).reshape(-1, 1)
y = np.array(list(lis[2].values())).reshape(-1, 1)
lr.fit(x,y)
y_pred = lr.predict(x)
plt.title("Year vs Number of Property Damage Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Property Damage Crashes")
plt.scatter(list(lis[2].keys()), list(lis[2].values()))
plt.plot(x, y_pred)
plt.show()
```


    
![png](output_45_0.png)
    



```python
print(lr.score(x,y))
print(lr.predict([[2022]]))
```

    0.3602981152383333
    [[1453650.66666667]]


With the Linear Regression, the predicted number of Property Damage Crashes are approximately 1,453,650 crashes, with an R^2 score of 0.36. The prediction score is a bit low, but it is following the trend of the overall data, which is decreasing. Let's see what a Quadration Regression provides us.


```python
poly = PolynomialFeatures(degree=2)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Property Damage Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Property Damage Crashes")
plt.scatter(list(lis[2].keys()), list(lis[2].values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_48_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.3645539132593628
    [[1477083.99999809]]


With the Quadratic Regression, the predicted number of Property Damage Crashes are approximately 1,477,083 crashes, with an R^2 score of 0.36. These values are similar to the values from the Linear Regression, so let's see what a Cubic Regression provides us.


```python
poly = PolynomialFeatures(degree=3)
poly_x = poly.fit_transform(x)
lr.fit(poly_x, y)
poly_ypred = lr.predict(poly_x)
plt.title("Year vs Number of Property Damage Crashes")
plt.xlabel("Year")
plt.ylabel("Number of Property Damage Crashes")
plt.scatter(list(lis[2].keys()), list(lis[2].values()))
plt.plot(x, poly_ypred)
plt.show()
```


    
![png](output_51_0.png)
    



```python
print(lr.score(poly_x,y))
print(lr.predict(poly.fit_transform([[2022]])))
```

    0.45170894227768355
    [[1695497.296875]]


With the Cubic Regression, the predicted number of property damage crashes are 1,695,496 crashes, with an R^2 score of 0.45. While the R^2 score is more accurate compared to the previous two regressions, the predicted number of crashes is much higher compared to the number of crashes in 2021, which is between 1,550,000-1,600,000 crashes. The overall trend of the data is decreasing. Therefore, with the Cubic Regression, we are seeing a case of overfitting. Since the values of the R^2 score and the predicted crashes in 2022 for the Linear and Quadratic regressions are similar to each other, it is safe to go with the Linear Model in order to prevent overfitting. Therefore, the Linear Regression is the accurate model with a predicted value of approximately 1,453,650 crashes.


```python
#-road condition vs number of crashes for PG, Moco, Batimore, Baltimore City in 2021
```


```python

county_year_road_group = df.groupby(["rd_cond_desc", "county_desc", "year"])
road_group = df.groupby([ "rd_cond_desc"])
x_axis = [k for k, v in road_group if (k != "No Defects") & (k != "Not Applicable")]
q_PG = []
q_Bmore = []
q_Bmore_City = []
q_MoCo = []
for  k, v in county_year_road_group:
  if (k[2] == 2021) & (k[0] != "No Defects" ) & (k[0] != "Not Applicable"):
    if k[1] == "Prince George's":
        q_PG.append((len(v.index), k[0]))
    if k[1] == "Montgomery":
      q_MoCo.append((len(v.index), k[0]))
    if k[1] == "Baltimore":
      q_Bmore.append((len(v.index), k[0]))
    if k[1] == "Baltimore City":
      q_Bmore_City.append((len(v.index), k[0]))

if len(x_axis) != len(q_PG):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_PG[i][1]:
        q_PG.insert(i, "0")
q_PG = [x[0] for x in q_PG]
q_PG = [int(x) for x in q_PG]

if len(x_axis) != len(q_Bmore):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_Bmore[i][1]:
        q_Bmore.insert(i, "0")
q_Bmore = [x[0] for x in q_Bmore]
q_Bmore = [int(x) for x in q_Bmore]

if len(x_axis) != len(q_Bmore_City):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_Bmore_City[i][1]:
        q_Bmore_City.insert(i, "0")
q_Bmore_City = [x[0] for x in q_Bmore_City]
q_Bmore_City = [int(x) for x in q_Bmore_City]


if len(x_axis) != len(q_MoCo):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_MoCo[i][1]:
        q_MoCo.insert(i, "0")

q_MoCo = [x[0] for x in q_MoCo]
q_MoCo = [int(x) for x in q_MoCo]



```


```python
fig = plt.figure()
plt.bar(x_axis, q_PG)
plt.xlabel('Road Condition')
plt.ylabel('Number of Crashes')
plt.title('Road Condition vs. Number of Crashes in Prince George\'s County')
fig.subplots_adjust(right = 3.5)
plt.show()

fig = plt.figure()
plt.bar(x_axis, q_Bmore)
plt.xlabel('Road Condition')
plt.ylabel('Number of Crashes')
plt.title('Road Condition vs. Number of Crashes in Baltimore County')
fig.subplots_adjust(right = 3.5)
plt.show()

fig = plt.figure()
plt.bar(x_axis, q_Bmore_City)
plt.xlabel('Road Condition')
plt.ylabel('Number of Crashes')
plt.title('Road Condition vs. Number of Crashes in Baltimore City')
fig.subplots_adjust(right = 3.5)
plt.show()

fig = plt.figure()
plt.bar(x_axis, q_MoCo)
plt.xlabel('Road Condition')
plt.ylabel('Number of Crashes')
plt.title('Road Condition vs. Number of Crashes in Montgomery County')
fig.subplots_adjust(right = 3.5)
plt.show()

```


    
![png](output_56_0.png)
    



    
![png](output_56_1.png)
    



    
![png](output_56_2.png)
    



    
![png](output_56_3.png)
    


Map for 

#-light condition vs number of crashes for PG, Moco, Batimore, Baltimore City in 2021


```python
light_year_road_group = df.groupby(["light_desc", "county_desc", "year"])
light_group = df.groupby([ "light_desc"])
x_axis = [k for k, v in light_group if (k != "Not Applicable")]
q_PG = []
q_Bmore = []
q_Bmore_City = []
q_MoCo = []

for  k, v in light_year_road_group:
  if (k[2] == 2021) & (k[0] != "No Defects" ) & (k[0] != "Not Applicable"):
    if k[1] == "Prince George's":
        q_PG.append((len(v.index), k[0]))
    if k[1] == "Montgomery":
      q_MoCo.append((len(v.index), k[0]))
    if k[1] == "Baltimore":
      q_Bmore.append((len(v.index), k[0]))
    if k[1] == "Baltimore City":
      q_Bmore_City.append((len(v.index), k[0]))

if len(x_axis) != len(q_PG):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_PG[i][1]:
        q_PG.insert(i, "0")
q_PG = [x[0] for x in q_PG]
q_PG = [int(x) for x in q_PG]

if len(x_axis) != len(q_Bmore):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_Bmore[i][1]:
        q_Bmore.insert(i, "0")
q_Bmore = [x[0] for x in q_Bmore]
q_Bmore = [int(x) for x in q_Bmore]

if len(x_axis) != len(q_Bmore_City):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_Bmore_City[i][1]:
        q_Bmore_City.insert(i, "0")
q_Bmore_City = [x[0] for x in q_Bmore_City]
q_Bmore_City = [int(x) for x in q_Bmore_City]

if len(x_axis) != len(q_MoCo):
    for i in range(0, len(x_axis)):
      if x_axis[i] != q_MoCo[i][1]:
        q_MoCo.insert(i, "0")
q_MoCo = [x[0] for x in q_MoCo]
q_MoCo = [int(x) for x in q_MoCo]


```


```python
fig = plt.figure()
plt.bar(x_axis, q_PG)
plt.xlabel('Light Condition')
plt.ylabel('Number of Crashes')
plt.title('Light Condition vs. Number of Crashes in Prince George\'s County')
fig.subplots_adjust(right = 3)
plt.show()

fig = plt.figure()
plt.bar(x_axis, q_Bmore)
plt.xlabel('Light Condition')
plt.ylabel('Number of Crashes')
plt.title('Light Condition vs. Number of Crashes in Baltimore County')
fig.subplots_adjust(right = 3)
plt.show()

fig = plt.figure()
plt.bar(x_axis, q_Bmore_City)
plt.xlabel('Light Condition')
plt.ylabel('Number of Crashes')
plt.title('Light Condition vs. Number of Crashes in Baltimore City')
fig.subplots_adjust(right = 3)
plt.show()

fig = plt.figure()
plt.bar(x_axis, q_MoCo)
plt.xlabel('Light Condition')
plt.ylabel('Number of Crashes')
plt.title('Light Condition vs. Number of Crashes in Montgomery County')
fig.subplots_adjust(right = 3)
plt.show()

```


    
![png](output_60_0.png)
    



    
![png](output_60_1.png)
    



    
![png](output_60_2.png)
    



    
![png](output_60_3.png)
    


road condition vs number of crashes for each county in 2021


```python
# counties = []
# for i in df["COUNTY_DESC"]:
#     if i not in counties:
#         counties.append(i)


# for county in counties:
#     bl = df[df["COUNTY_DESC"] == county ]
#     bl = bl.dropna(subset=["RD_COND_DESC"])
#     data = dict(Counter(bl["RD_COND_DESC"]))
#     data.pop("No Defects")
#     data.pop("Not Applicable")
#     data.pop("Unknown")
#     values = data.values()
#     keys = list(data.keys())
#     plt.bar(range(len(data)), values, tick_label=keys)
#     plt.title(county)
#     plt.xticks(fontsize=14, rotation=90)
#     plt.show()
```

type of light_desc vs number of crashes for each county in 2021


```python
# for county in counties:
#     bl = df[df["COUNTY_DESC"] == county]
#     bl = bl.dropna(subset=["LIGHT_DESC"])
#     data = dict(Counter(bl["LIGHT_DESC"]))
#     data.pop("Other")
#     data.pop("Not Applicable")
#     data.pop("Unknown")
#     values = data.values()
#     keys = list(data.keys())
#     plt.bar(range(len(data)), values, tick_label=keys)
#     plt.title(county)
#     plt.xticks(fontsize=14, rotation=90)
#     plt.show()
```

#4. Conclusion and further solution

Now we will show some maps of 4 areas in maryland. In each area we will mark the collisions based on road condition and light descrition. The four areas we will be looking at are Prince George's county, Baltimore County, Montgomery County, and Baltimore City.

First lets start with the light descrition of all 4 counties in quarter 4 of 2021.

**Prince George's county**


```python
map_osm1 = folium.Map(location=[38.7849, -76.8721], zoom_start=11.8)



count1 = count2 = count3 = count4 = count5 = 0
for index, crash in df[df["quarter"] == "Q4"].iterrows():
    if (crash["county_desc"] == "Prince George's") & (crash["light_desc"] == "Dark Lights On") & (crash["year"] == 2021) & (count1 <= 20) :
        count1 = count1 + 1
        folium.Marker(location=[crash["latitude"], crash["longitude"]],
                    icon=folium.Icon(color='red')).add_to(map_osm1)

    if (crash["county_desc"] == "Prince George's") & (crash["light_desc"] == "Dark No Lights") & (crash["year"] == 2021) & (count2 <= 20) :
        count2 = count2 + 1
        folium.Marker(location=[crash["latitude"], crash["longitude"]],
                    icon=folium.Icon(color='lightblue')).add_to(map_osm1)

    if (crash["county_desc"] == "Prince George's") & (crash["light_desc"] == "Daylight") & (crash["year"] == 2021) & (count3 <= 20) :
        count3 = count3 + 1
        folium.Marker(location=[crash["latitude"], crash["longitude"]],
                    icon=folium.Icon(color='orange')).add_to(map_osm1) #A LOT OF FAYLIGHT CRASHES IN MUDDLE OF BMORE CITY

    if (crash["county_desc"] == "Prince George's") & (crash["light_desc"] == "Dawn") & (crash["year"] == 2021) & (count4 <= 20) :
        count4 = count4 + 1
        folium.Marker(location=[crash["latitude"], crash["longitude"]],
                    icon=folium.Icon(color='black')).add_to(map_osm1)

    if (crash["county_desc"] == "Prince George's") & (crash["light_desc"] == "Dusk") & (crash["year"] == 2021) & (count5 <= 20) :
        count5 = count5 + 1
        folium.Marker(location=[crash["latitude"], crash["longitude"]],
                    icon=folium.Icon(color='green')).add_to(map_osm1)

map_osm1
```


    ---------------------------------------------------------------------------

    RecursionError                            Traceback (most recent call last)

    <ipython-input-40-f66a8beccefa> in <module>()
         18         count3 = count3 + 1
         19         folium.Marker(location=[crash["latitude"], crash["longitude"]],
    ---> 20                     icon=folium.Icon(color='orange')).add_to(map_osm1) #A LOT OF FAYLIGHT CRASHES IN MUDDLE OF BMORE CITY
         21 
         22     if (crash["county_desc"] == "Prince George's") & (crash["light_desc"] == "Dawn") & (crash["year"] == 2021) & (count4 <= 20) :


    /usr/local/lib/python3.7/dist-packages/folium/map.py in __init__(self, location, popup, tooltip, icon, draggable)
        256         super(Marker, self).__init__()
        257         self._name = 'Marker'
    --> 258         self.location = _validate_coordinates(location)
        259         self.draggable = draggable
        260         if icon is not None:


    /usr/local/lib/python3.7/dist-packages/folium/utilities.py in _validate_coordinates(coordinates)
         51 def _validate_coordinates(coordinates):
         52     """Validates multiple coordinates for the various markers in folium."""
    ---> 53     if _isnan(coordinates):
         54         raise ValueError('Location values cannot contain NaNs, '
         55                          'got:\n{!r}'.format(coordinates))


    /usr/local/lib/python3.7/dist-packages/folium/utilities.py in _isnan(values)
         77 def _isnan(values):
         78     """Check if there are NaNs values in the iterable."""
    ---> 79     return any(math.isnan(value) for value in _flatten(values))
         80 
         81 


    /usr/local/lib/python3.7/dist-packages/folium/utilities.py in <genexpr>(.0)
         77 def _isnan(values):
         78     """Check if there are NaNs values in the iterable."""
    ---> 79     return any(math.isnan(value) for value in _flatten(values))
         80 
         81 


    /usr/local/lib/python3.7/dist-packages/folium/utilities.py in _flatten(container)
         69     for i in container:
         70         if _is_sized_iterable(i):
    ---> 71             for j in _flatten(i):
         72                 yield j
         73         else:


    ... last 1 frames repeated, from the frame below ...


    /usr/local/lib/python3.7/dist-packages/folium/utilities.py in _flatten(container)
         69     for i in container:
         70         if _is_sized_iterable(i):
    ---> 71             for j in _flatten(i):
         72                 yield j
         73         else:


    RecursionError: maximum recursion depth exceeded in comparison

