---
layout: post
title:  "Identify Fraud from Enron Email"
date:   2016-07-05 13:00:00 +0900
categories: udacity en
---

Once successful and dominant company in the energy business, but has fallen because they tried to hide company losses and avoid taxes by creating made-up entities, known since as the Enron scandal. The most infamous name associated with this scandal is Kenneth Lay, who is on trial as of januray 2006. he is charged on 11 counts ranging from insider trading to bank fraud. 
<!--more-->
The goal of this project is to identify Enron employees who may have committed fraud based on the public Enron financial and email dataset. To find out person of interest, we can explore many features and apply specific algorithms. Also, we should validate the performance by evaluating metircs. Through this process, we can gain useful insights from dataset.

- <a href='#1.-Data-exploration'>1. Data exploration</a>
- <a href='#2.-Outlier-Investigation'>2. Outlier Investigation</a>
- <a href='#3.-Feature-Selection'>3. Feature Selection</a>
- <a href='#4.-Algorithms'>4. Algorithms</a>
- <a href='#5.-Evaluation'>5. Evaluation</a>
- <a href='#Appendix'>Appendix</a>

## 1. Data exploration

I looked through dataset to describe overall statistics and find out what features can help identifying fraud from Enron email. I have financial and email dataset labeled who is POI. With 21 features, I categorized list of features as following:
<table border="1">
  <tr>
    <th rowspan="2">Label</th>
    <th colspan="2">Financial</th>
    <th rowspan="2">Email (6 features)</th>
  </tr>
  <tr>
    <td>payments (10 features)</td>
    <td>stock value (4 features)</td>
  </tr>
  <tr>
    <td><li>persons-of-interest</li></td>
    <td>
<ul>
<li>salary</li>
<li>bonus</li>
<li>long_term_incentive</li>
<li>deferral_payments</li>
<li>expenses</li>
<li>deferred_income</li>
<li>director_fees</li>
<li>loan_advances</li>
<li>other</li>
<li>total_payments</li>
</ul>
</td>
    <td>
<ul>
<li>restricted_stock</li>
<li>exercised_stock_options</li>
<li>restricted_stock_deferred</li>
<li>total_stock_value</li>
</ul>
</td>
    <td>
    <ul><li>from_messages</li>
<li>to_messages</li>
<li>shared_receipt_with_poi</li>
<li>from_this_person_to_poi</li>
<li>email_address</li>
<li>from_poi_to_this_person</li></ul>
</td>
  </tr>
</table>

The first 5 rows of dataset:



```python
# %run ./exploration.py
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from tester import test_classifier, dump_classifier_and_data

features_list = ['poi', 'salary', 'bonus','long_term_incentive','deferral_payments',
                 'expenses','deferred_income','director_fees','loan_advances','other',
                 'restricted_stock', 'exercised_stock_options','restricted_stock_deferred', 
               'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
import pandas as pd
import numpy as np

df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
df.reset_index(level=0, inplace=True)
columns = list(df.columns)
columns[0] = 'name'
df.columns = columns
df.fillna(0, inplace=True)
df.head()


```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>bonus</th>
      <th>restricted_stock</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock_deferred</th>
      <th>...</th>
      <th>loan_advances</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>director_fees</th>
      <th>deferred_income</th>
      <th>long_term_incentive</th>
      <th>email_address</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALLEN PHILLIP K</td>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>1729541</td>
      <td>4175000</td>
      <td>126027</td>
      <td>1407</td>
      <td>-126027</td>
      <td>...</td>
      <td>0</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>-3081055</td>
      <td>304805</td>
      <td>phillip.allen@enron.com</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BADUM JAMES P</td>
      <td>0</td>
      <td>0</td>
      <td>178980</td>
      <td>182466</td>
      <td>257817</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BANNANTINE JAMES M</td>
      <td>477</td>
      <td>566</td>
      <td>0</td>
      <td>916197</td>
      <td>4046157</td>
      <td>0</td>
      <td>1757552</td>
      <td>465</td>
      <td>-560222</td>
      <td>...</td>
      <td>0</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-5104</td>
      <td>0</td>
      <td>james.bannantine@enron.com</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BAXTER JOHN C</td>
      <td>267102</td>
      <td>0</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>6680544</td>
      <td>1200000</td>
      <td>3942714</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>2660303</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1386055</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAY FRANKLIN R</td>
      <td>239671</td>
      <td>0</td>
      <td>260455</td>
      <td>827696</td>
      <td>0</td>
      <td>400000</td>
      <td>145796</td>
      <td>0</td>
      <td>-82782</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-201641</td>
      <td>0</td>
      <td>frank.bay@enron.com</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Overall statistics of dataset:


```python
df.describe().transpose()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bonus</th>
      <td>146</td>
      <td>1333474.232877</td>
      <td>8094029.239637</td>
      <td>0</td>
      <td>0.00</td>
      <td>300000.0</td>
      <td>800000.00</td>
      <td>97343619</td>
    </tr>
    <tr>
      <th>deferral_payments</th>
      <td>146</td>
      <td>438796.520548</td>
      <td>2741325.337926</td>
      <td>-102500</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>9684.50</td>
      <td>32083396</td>
    </tr>
    <tr>
      <th>deferred_income</th>
      <td>146</td>
      <td>-382762.205479</td>
      <td>2378249.890202</td>
      <td>-27992891</td>
      <td>-37926.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>director_fees</th>
      <td>146</td>
      <td>19422.486301</td>
      <td>119054.261157</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1398517</td>
    </tr>
    <tr>
      <th>email_address</th>
      <td>146</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>exercised_stock_options</th>
      <td>146</td>
      <td>4182736.198630</td>
      <td>26070399.807568</td>
      <td>0</td>
      <td>0.00</td>
      <td>608293.5</td>
      <td>1714220.75</td>
      <td>311764000</td>
    </tr>
    <tr>
      <th>expenses</th>
      <td>146</td>
      <td>70748.267123</td>
      <td>432716.319438</td>
      <td>0</td>
      <td>0.00</td>
      <td>20182.0</td>
      <td>53740.75</td>
      <td>5235198</td>
    </tr>
    <tr>
      <th>from_messages</th>
      <td>146</td>
      <td>358.602740</td>
      <td>1441.259868</td>
      <td>0</td>
      <td>0.00</td>
      <td>16.5</td>
      <td>51.25</td>
      <td>14368</td>
    </tr>
    <tr>
      <th>from_poi_to_this_person</th>
      <td>146</td>
      <td>38.226027</td>
      <td>73.901124</td>
      <td>0</td>
      <td>0.00</td>
      <td>2.5</td>
      <td>40.75</td>
      <td>528</td>
    </tr>
    <tr>
      <th>from_this_person_to_poi</th>
      <td>146</td>
      <td>24.287671</td>
      <td>79.278206</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>13.75</td>
      <td>609</td>
    </tr>
    <tr>
      <th>loan_advances</th>
      <td>146</td>
      <td>1149657.534247</td>
      <td>9649342.029695</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>83925000</td>
    </tr>
    <tr>
      <th>long_term_incentive</th>
      <td>146</td>
      <td>664683.945205</td>
      <td>4046071.990875</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>375064.75</td>
      <td>48521928</td>
    </tr>
    <tr>
      <th>other</th>
      <td>146</td>
      <td>585431.794521</td>
      <td>3682344.576631</td>
      <td>0</td>
      <td>0.00</td>
      <td>959.5</td>
      <td>150606.50</td>
      <td>42667589</td>
    </tr>
    <tr>
      <th>poi</th>
      <td>146</td>
      <td>0.123288</td>
      <td>0.329899</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>restricted_stock</th>
      <td>146</td>
      <td>1749257.020548</td>
      <td>10899953.192164</td>
      <td>-2604490</td>
      <td>8115.00</td>
      <td>360528.0</td>
      <td>814528.00</td>
      <td>130322299</td>
    </tr>
    <tr>
      <th>restricted_stock_deferred</th>
      <td>146</td>
      <td>20516.369863</td>
      <td>1439660.966040</td>
      <td>-7576788</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>15456290</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>146</td>
      <td>365811.356164</td>
      <td>2203574.963717</td>
      <td>0</td>
      <td>0.00</td>
      <td>210596.0</td>
      <td>270850.50</td>
      <td>26704229</td>
    </tr>
    <tr>
      <th>shared_receipt_with_poi</th>
      <td>146</td>
      <td>692.986301</td>
      <td>1072.969492</td>
      <td>0</td>
      <td>0.00</td>
      <td>102.5</td>
      <td>893.50</td>
      <td>5521</td>
    </tr>
    <tr>
      <th>to_messages</th>
      <td>146</td>
      <td>1221.589041</td>
      <td>2226.770637</td>
      <td>0</td>
      <td>0.00</td>
      <td>289.0</td>
      <td>1585.75</td>
      <td>15149</td>
    </tr>
    <tr>
      <th>total_payments</th>
      <td>146</td>
      <td>4350621.993151</td>
      <td>26934479.950729</td>
      <td>0</td>
      <td>93944.75</td>
      <td>941359.5</td>
      <td>1968286.75</td>
      <td>309886585</td>
    </tr>
    <tr>
      <th>total_stock_value</th>
      <td>146</td>
      <td>5846018.075342</td>
      <td>36246809.190047</td>
      <td>-44093</td>
      <td>228869.50</td>
      <td>965955.0</td>
      <td>2319991.25</td>
      <td>434509511</td>
    </tr>
  </tbody>
</table>
</div>



Number of person-of-interest and non person-of-interest:


```python
bypoi = df.groupby(['poi'])
print bypoi['poi'].aggregate([len])

```

         len
    poi     
    0    128
    1     18


<div> </div>

## 2. Outlier Investigation

There was one outlier and two correction in the dataset.
- outlier : datapoint of salary 26,704,229 is clear outlier. The sum of all salaries seem to be parsed accidently from pdf file, so I deleted it.


```python
del data_dict['TOTAL']
```

- incorrect numbers : BELFER ROBERT, BHATNAGAR SANJAY records were incorrect, so I updated value according to Enron Statement of Financial Affairs pdf file.


```python
payment_cols = ['salary', 'bonus','long_term_incentive','deferral_payments','expenses','deferred_income','director_fees','loan_advances','other']
stock_cols = ['restricted_stock', 'exercised_stock_options','restricted_stock_deferred']
def check_consistency(df):
    consistency = pd.DataFrame()
    consistency['name'] = df['name']
    consistency['total1'] = df[payment_cols].sum(axis=1)
    consistency['total2'] = df[stock_cols].sum(axis=1)
    consistency['consistent_payments'] = (consistency['total1'] == df['total_payments'])
    consistency['consistent_stockvalue'] = (consistency['total2'] == df['total_stock_value'])
    checks = consistency[(consistency['consistent_payments'] == False) | (consistency['consistent_stockvalue'] == False)]['name'].tolist()

    return checks
    

check_names = check_consistency(df)

print check_names
#if len(check_names) > 0:
payment_cols.append('total_payments')
df[df['name'].isin(check_names)][payment_cols]
```

    ['BELFER ROBERT', 'BHATNAGAR SANJAY']





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferral_payments</th>
      <th>expenses</th>
      <th>deferred_income</th>
      <th>director_fees</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>total_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-102500</td>
      <td>0</td>
      <td>0</td>
      <td>3285</td>
      <td>0</td>
      <td>0</td>
      <td>102500</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>137864</td>
      <td>0</td>
      <td>137864</td>
      <td>15456290</td>
    </tr>
  </tbody>
</table>
</div>




```python
#if len(check_names) > 0:
stock_cols.append('total_stock_value')
df[df['name'].isin(check_names)][stock_cols]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>restricted_stock</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>3285</td>
      <td>44093</td>
      <td>-44093</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-2604490</td>
      <td>2604490</td>
      <td>15456290</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
payment_cols.remove('total_payments')
stock_cols.remove('total_stock_value')

data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285

data_dict['BHATNAGAR SANJAY']['total_payments'] =137864
data_dict['BHATNAGAR SANJAY']['expenses'] =137864
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0

data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 0

data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
```

## 3. Feature Selection

- ### new features (fraction_from_poi, fraction_to_poi)
    I created new features: fraction_from_poi, fraction_to_poi. **Number of messages related to POI are divided by the total number of messages to or from this person**. 
    The figure on the left showed scatter plot of old features that have sparse POIs(red points). On the other hand, POIs(red points) of the left figure are denser. 
    


```python
def compute_fraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    import math
    if poi_messages == 0 or all_messages == 0 or math.isnan(float(poi_messages)) or math.isnan(float(all_messages)) :
        return 0.
    fraction = 0.
    fraction = float(poi_messages) / float(all_messages) 
    return fraction

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = round(fraction_from_poi,3)
    
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction( from_this_person_to_poi, from_messages )
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = round(fraction_to_poi, 3)

    ## append two features to the list    
if not ('fraction_from_poi' in set(features_list)):
    features_list.append('fraction_from_poi')
if not ('fraction_to_poi' in set(features_list)):
    features_list.append('fraction_to_poi')

%pylab inline

import matplotlib.pyplot as plt
 
def graph_scatter_with_poi(var1, var2):
    for name in data_dict:
        point = data_dict[name]
        poi = point['poi']
        x = point[var1]
        y = point[var2]

        if poi:
            plt.scatter( x, y, color='red')
        else:
            plt.scatter( x, y, color='blue')
    plt.xlabel(var1)
    plt.ylabel(var2)

plt.figure(1, figsize=(16, 5))
plt.subplot(1,2,1) 
graph_scatter_with_poi('from_poi_to_this_person', 'from_this_person_to_poi')
plt.subplot(1,2,2) 
graph_scatter_with_poi('fraction_from_poi', 'fraction_to_poi')
```

    Populating the interactive namespace from numpy and matplotlib


    WARNING: pylab import has clobbered these variables: ['random']
    `%matplotlib` prevents importing * from pylab and numpy



![png](p5_17_2.png)


An example of employee with new features:


```python
data_dict['SKILLING JEFFREY K']
```




    {'bonus': 5600000,
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'jeff.skilling@enron.com',
     'exercised_stock_options': 19250000,
     'expenses': 29336,
     'fraction_from_poi': 0.024,
     'fraction_to_poi': 0.278,
     'from_messages': 108,
     'from_poi_to_this_person': 88,
     'from_this_person_to_poi': 30,
     'loan_advances': 'NaN',
     'long_term_incentive': 1920000,
     'other': 22122,
     'poi': True,
     'restricted_stock': 6843672,
     'restricted_stock_deferred': 'NaN',
     'salary': 1111258,
     'shared_receipt_with_poi': 2042,
     'to_messages': 3627,
     'total_payments': 8682716,
     'total_stock_value': 26093672}



### intelligently select feature


- automated feature selection

I transformed features with feature scaling(MinMaxScaler), feature selection(SelectKBest).
SelectKBest was used in chapter 4.


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from tester import dump_classifier_and_data

from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pprint


### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

folds = 1000
random = 13
cv = StratifiedShuffleSplit(labels, folds, random_state=random)
mdf = []

combined_features = FeatureUnion( [
            ('scaler', MinMaxScaler())
        ])

```

I did not include to_messages, from_messages in the features_list. Because new features, fraction_from_poi and fraction_to_poi, are more relevant to predict POI, as shown in the table.


```python
features_list_fs = list(features_list)
if not ('to_messages' in set(features_list_fs)):
    features_list_fs.append('to_messages')
if not ('from_messages' in set(features_list_fs)):
    features_list_fs.append('from_messages')

data_fs = featureFormat(my_dataset, features_list_fs, sort_keys = True)
labels_fs, features_fs = targetFeatureSplit(data_fs)

pipeline = Pipeline([
        ("features", combined_features), 
        ('kbest', SelectKBest(k='all', score_func=f_classif)),
        ('DecisionTree', DecisionTreeClassifier(random_state=random, min_samples_split=20, criterion='entropy', max_features=None))
        ])

pipeline.fit(features_fs, labels_fs)

scores = pipeline.named_steps['kbest'].scores_
df = pd.DataFrame(data = list(zip(features_list_fs[1:], scores)), columns=['Feature', 'Score'])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>salary</td>
      <td>18.575703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bonus</td>
      <td>21.060002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>long_term_incentive</td>
      <td>10.072455</td>
    </tr>
    <tr>
      <th>3</th>
      <td>deferral_payments</td>
      <td>0.221214</td>
    </tr>
    <tr>
      <th>4</th>
      <td>expenses</td>
      <td>5.550684</td>
    </tr>
    <tr>
      <th>5</th>
      <td>deferred_income</td>
      <td>11.561888</td>
    </tr>
    <tr>
      <th>6</th>
      <td>director_fees</td>
      <td>2.112762</td>
    </tr>
    <tr>
      <th>7</th>
      <td>loan_advances</td>
      <td>7.242730</td>
    </tr>
    <tr>
      <th>8</th>
      <td>other</td>
      <td>4.219888</td>
    </tr>
    <tr>
      <th>9</th>
      <td>restricted_stock</td>
      <td>8.958540</td>
    </tr>
    <tr>
      <th>10</th>
      <td>exercised_stock_options</td>
      <td>22.610531</td>
    </tr>
    <tr>
      <th>11</th>
      <td>restricted_stock_deferred</td>
      <td>0.761863</td>
    </tr>
    <tr>
      <th>12</th>
      <td>shared_receipt_with_poi</td>
      <td>8.746486</td>
    </tr>
    <tr>
      <th>13</th>
      <td>fraction_from_poi</td>
      <td>3.230112</td>
    </tr>
    <tr>
      <th>14</th>
      <td>fraction_to_poi</td>
      <td>16.642573</td>
    </tr>
    <tr>
      <th>15</th>
      <td>to_messages</td>
      <td>1.698824</td>
    </tr>
    <tr>
      <th>16</th>
      <td>from_messages</td>
      <td>0.164164</td>
    </tr>
  </tbody>
</table>
</div>



Also, I confirmed that precision and recall were higher without original features. This means, without these features, I have more chances to find out real POI, and less chances that non-POIs get flagged.

With to_messages, from_messages:
    - Accuracy: 0.86073	Precision: 0.47707	Recall: 0.46300	F1: 0.46993	F2: 0.46575
    
Without to_messages, from_messages:
    - Accuracy: 0.85860	Precision: 0.46870	Recall: 0.45300	F1: 0.46072	F2: 0.45606


## 4. Algorithms

I feed transformed features to each algorithm in the following. 
- SVC (svm)
- **DecisionTreeClassifier** (tree)
- GaussianNB (naive bayes)


### SVM


```python
pipeline = Pipeline([("features", combined_features), ('svc', SVC())])

param_grid = {
    'svc__kernel': [ 'sigmoid', 'poly','rbf'],
    #'svc__C': [0.1, 1, 10],
    'svc__gamma': ['auto'],
    'svc__class_weight' :[None, 'balanced']
      }

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1', verbose=1) # f1 for binary targets
grid_search.fit(features, labels)
print grid_search.best_score_
print grid_search.best_params_
```

    [Parallel(n_jobs=1)]: Done  49 tasks       | elapsed:    0.2s
    [Parallel(n_jobs=1)]: Done 199 tasks       | elapsed:    0.8s
    [Parallel(n_jobs=1)]: Done 449 tasks       | elapsed:    1.8s
    [Parallel(n_jobs=1)]: Done 799 tasks       | elapsed:    3.2s
    [Parallel(n_jobs=1)]: Done 1249 tasks       | elapsed:    5.1s
    [Parallel(n_jobs=1)]: Done 1799 tasks       | elapsed:    7.4s
    [Parallel(n_jobs=1)]: Done 2449 tasks       | elapsed:   10.0s
    [Parallel(n_jobs=1)]: Done 3199 tasks       | elapsed:   13.0s
    [Parallel(n_jobs=1)]: Done 4049 tasks       | elapsed:   16.7s
    [Parallel(n_jobs=1)]: Done 4999 tasks       | elapsed:   20.9s


    Fitting 1000 folds for each of 6 candidates, totalling 6000 fits
    0.419783910534
    {'svc__gamma': 'auto', 'svc__class_weight': 'balanced', 'svc__kernel': 'rbf'}


    [Parallel(n_jobs=1)]: Done 6000 out of 6000 | elapsed:   25.4s finished



```python
clf_fin = pipeline.set_params(**grid_search.best_params_)
test_classifier(clf_fin, my_dataset, features_list)
```

    Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))],
           transformer_weights=None)), ('svc', SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])
    	Accuracy: 0.80380	Precision: 0.34893	Recall: 0.54450	F1: 0.42531	F2: 0.48961
    	Total predictions: 15000	True positives: 1089	False positives: 2032	False negatives:  911	True negatives: 10968
    


### DecisionTreeClassifier


```python
pipeline = Pipeline([("features", combined_features), ('DecisionTree', DecisionTreeClassifier(random_state=random))])

param_grid = {
    'DecisionTree__min_samples_split':[20, 30, 40],
    'DecisionTree__max_features': [None, 'auto', 'log2'],
    'DecisionTree__criterion': ['gini', 'entropy']
      }

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1', verbose=1) # f1 for binary targets
grid_search.fit(features, labels)
print grid_search.best_score_
print grid_search.best_params_

```

    [Parallel(n_jobs=1)]: Done  49 tasks       | elapsed:    0.2s
    [Parallel(n_jobs=1)]: Done 199 tasks       | elapsed:    0.7s
    [Parallel(n_jobs=1)]: Done 449 tasks       | elapsed:    1.6s
    [Parallel(n_jobs=1)]: Done 799 tasks       | elapsed:    2.8s
    [Parallel(n_jobs=1)]: Done 1249 tasks       | elapsed:    4.4s
    [Parallel(n_jobs=1)]: Done 1799 tasks       | elapsed:    6.4s
    [Parallel(n_jobs=1)]: Done 2449 tasks       | elapsed:    9.0s
    [Parallel(n_jobs=1)]: Done 3199 tasks       | elapsed:   11.6s
    [Parallel(n_jobs=1)]: Done 4049 tasks       | elapsed:   14.5s
    [Parallel(n_jobs=1)]: Done 4999 tasks       | elapsed:   17.7s
    [Parallel(n_jobs=1)]: Done 6049 tasks       | elapsed:   21.3s
    [Parallel(n_jobs=1)]: Done 7199 tasks       | elapsed:   25.2s
    [Parallel(n_jobs=1)]: Done 8449 tasks       | elapsed:   29.4s
    [Parallel(n_jobs=1)]: Done 9799 tasks       | elapsed:   34.2s
    [Parallel(n_jobs=1)]: Done 11249 tasks       | elapsed:   39.5s
    [Parallel(n_jobs=1)]: Done 12799 tasks       | elapsed:   45.1s
    [Parallel(n_jobs=1)]: Done 14449 tasks       | elapsed:   50.7s
    [Parallel(n_jobs=1)]: Done 16199 tasks       | elapsed:   56.8s


    Fitting 1000 folds for each of 18 candidates, totalling 18000 fits
    0.427048412698
    {'DecisionTree__criterion': 'entropy', 'DecisionTree__min_samples_split': 20, 'DecisionTree__max_features': None}


    [Parallel(n_jobs=1)]: Done 18000 out of 18000 | elapsed:  1.1min finished



```python
fi = grid_search.best_estimator_.named_steps['DecisionTree'].feature_importances_ 

df = pd.DataFrame(data = list(zip(features_list[1:], fi)), columns=['Feature', 'Importance'])
df

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>salary</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bonus</td>
      <td>0.093646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>long_term_incentive</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>deferral_payments</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>expenses</td>
      <td>0.230564</td>
    </tr>
    <tr>
      <th>5</th>
      <td>deferred_income</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>director_fees</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>loan_advances</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>other</td>
      <td>0.484556</td>
    </tr>
    <tr>
      <th>9</th>
      <td>restricted_stock</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>exercised_stock_options</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>restricted_stock_deferred</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>shared_receipt_with_poi</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>fraction_from_poi</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>fraction_to_poi</td>
      <td>0.191234</td>
    </tr>
  </tbody>
</table>
</div>




```python
clf_fin = pipeline.set_params(**grid_search.best_params_)
test_classifier(clf_fin, my_dataset, features_list)
```

    Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))],
           transformer_weights=None)), ('DecisionTree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=20, min_weight_fraction_leaf=0.0,
                presort=False, random_state=13, splitter='best'))])
    	Accuracy: 0.86073	Precision: 0.47707	Recall: 0.46300	F1: 0.46993	F2: 0.46575
    	Total predictions: 15000	True positives:  926	False positives: 1015	False negatives: 1074	True negatives: 11985
    


### GaussianNB


```python


pipeline = Pipeline([("features", combined_features), ('GaussianNB', GaussianNB())])
param_grid = {
## no params
      }

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1', verbose=1) # f1 for binary targets
grid_search.fit(features, labels)
print grid_search.best_score_
print grid_search.best_params_
```

    [Parallel(n_jobs=1)]: Done  49 tasks       | elapsed:    0.1s
    [Parallel(n_jobs=1)]: Done 199 tasks       | elapsed:    0.4s
    [Parallel(n_jobs=1)]: Done 449 tasks       | elapsed:    0.8s
    [Parallel(n_jobs=1)]: Done 799 tasks       | elapsed:    1.5s


    Fitting 1000 folds for each of 1 candidates, totalling 1000 fits
    0.265048897508
    {}


    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:    1.8s finished



```python
clf_fin = pipeline.set_params(**grid_search.best_params_)
test_classifier(clf_fin, my_dataset, features_list)
```

    Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))],
           transformer_weights=None)), ('GaussianNB', GaussianNB())])
    	Accuracy: 0.34073	Precision: 0.15589	Recall: 0.89350	F1: 0.26547	F2: 0.45908
    	Total predictions: 15000	True positives: 1787	False positives: 9676	False negatives:  213	True negatives: 3324
    


### Overall result

SVM

    Accuracy: 0.80380	Precision: 0.34893	Recall: 0.54450	F1: 0.42531	F2: 0.48961
    Accuracy: 0.80380	Precision: 0.34893	Recall: 0.54450	F1: 0.42531	F2: 0.48961
    Accuracy: 0.80380	Precision: 0.34893	Recall: 0.54450	F1: 0.42531	F2: 0.48961
    
DecisionTreeClassifier

    Accuracy: 0.86120	Precision: 0.47880	Recall: 0.46300	F1: 0.47077	F2: 0.46608
    Accuracy: 0.86067	Precision: 0.47690	Recall: 0.46450	F1: 0.47062	F2: 0.46693
    Accuracy: 0.86093	Precision: 0.47790	Recall: 0.46500	F1: 0.47136	F2: 0.46752
    
GaussianNB

    Accuracy: 0.34073	Precision: 0.15589	Recall: 0.89350	F1: 0.26547	F2: 0.45908
    Accuracy: 0.34073	Precision: 0.15589	Recall: 0.89350	F1: 0.26547	F2: 0.45908
    Accuracy: 0.34073	Precision: 0.15589	Recall: 0.89350	F1: 0.26547	F2: 0.45908
    
    
I tested each algorithm three times to make sure their performances. As a result, **DecisionTreeClassifier** was the best.

### Tune the algorithm

Tuning is adjusting parameters of algorithm to improve performance. 
If I don't tune well, overfitting might occur. Even though it correctly classifies the data, its prediction can not be generalized.
I can contorl this problem through the parameter of an algorithm, so I tuned the paramethers with GridSearchCV. 

Since **DecisionTreeClassifier** showed the best performance,  I decided to tune additional parameters to improve performance.
I added feature selection(SelectKBest) in the pipeline, and I found that 12 parameters provide best score.


```python
%%time

pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kbest', SelectKBest()),
        ('dtree', DecisionTreeClassifier(random_state=random))])

param_grid = {              
    #'kbest__k':[1, 2, 3, 4, 5],
    #'kbest__k':[6,7,8,9,10],
    'kbest__k':[11,12,13,14,15],
    'dtree__max_features': [None, 'auto'],
    'dtree__criterion': ['entropy'],
    'dtree__max_depth': [None, 3, 5],
    'dtree__min_samples_split': [2, 1, 3],
    'dtree__min_samples_leaf': [1, 2],
    'dtree__min_weight_fraction_leaf': [0, 0.5],
    'dtree__class_weight': [{1: 1, 0: 1}, {1: 0.8, 0: 0.3}, {1:0.7, 0:0.4}],
    'dtree__splitter': ['best', 'random']
      }

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1', verbose=1) # f1 for binary targets
grid_search.fit(features, labels)
print grid_search.best_score_
print grid_search.best_params_

# k = 1,2,3,4,5
# Fitting 1000 folds for each of 2160 candidates, totalling 2160000 fits
# 0.36747950938
# {'dtree__min_samples_leaf': 2, 'dtree__min_samples_split': 2, 'kbest__k': 5, 'dtree__splitter': 'random', 
#'dtree__max_features': None, 'dtree__max_depth': 5, 'dtree__min_weight_fraction_leaf': 0, 
#'dtree__class_weight': {0: 0.3, 1: 0.8}, 'dtree__criterion': 'entropy'}
# CPU times: user 3h 15min 23s, sys: 8.36 s, total: 3h 15min 32s
# Wall time: 3h 15min 26s

# k = 6,7,8,9,10
# Fitting 1000 folds for each of 2160 candidates, totalling 2160000 fits
# 0.472646031746
# {'dtree__min_samples_leaf': 2, 'dtree__min_samples_split': 2, 'kbest__k': 10, 'dtree__splitter': 'random', 'dtree__max_features': 'auto', 
#'dtree__max_depth': 3, 'dtree__min_weight_fraction_leaf': 0, 'dtree__class_weight': {0: 0.3, 1: 0.8}, 'dtree__criterion': 'entropy'}
# CPU times: user 3h 16min 43s, sys: 8.48 s, total: 3h 16min 51s
# Wall time: 3h 16min 44s

```

    [Parallel(n_jobs=1)]: Done  49 tasks       | elapsed:    0.3s
    [Parallel(n_jobs=1)]: Done 199 tasks       | elapsed:    1.1s
    [Parallel(n_jobs=1)]: Done 449 tasks       | elapsed:    2.5s
    [Parallel(n_jobs=1)]: Done 799 tasks       | elapsed:    4.5s
    [Parallel(n_jobs=1)]: Done 1249 tasks       | elapsed:    6.9s
    [Parallel(n_jobs=1)]: Done 1799 tasks       | elapsed:    9.8s
    [Parallel(n_jobs=1)]: Done 2449 tasks       | elapsed:   13.2s
    [Parallel(n_jobs=1)]: Done 3199 tasks       | elapsed:   17.1s
    [Parallel(n_jobs=1)]: Done 4049 tasks       | elapsed:   21.6s
    [Parallel(n_jobs=1)]: Done 4999 tasks       | elapsed:   26.6s
    [Parallel(n_jobs=1)]: Done 6049 tasks       | elapsed:   31.8s
    [Parallel(n_jobs=1)]: Done 7199 tasks       | elapsed:   37.4s
    [Parallel(n_jobs=1)]: Done 8449 tasks       | elapsed:   43.5s
    [Parallel(n_jobs=1)]: Done 9799 tasks       | elapsed:   50.1s
    [Parallel(n_jobs=1)]: Done 11249 tasks       | elapsed:   57.1s
    [Parallel(n_jobs=1)]: Done 12799 tasks       | elapsed:  1.1min
    [Parallel(n_jobs=1)]: Done 14449 tasks       | elapsed:  1.2min
    [Parallel(n_jobs=1)]: Done 16199 tasks       | elapsed:  1.4min
    [Parallel(n_jobs=1)]: Done 18049 tasks       | elapsed:  1.5min
    [Parallel(n_jobs=1)]: Done 19999 tasks       | elapsed:  1.7min
    [Parallel(n_jobs=1)]: Done 22049 tasks       | elapsed:  1.8min
    [Parallel(n_jobs=1)]: Done 24199 tasks       | elapsed:  2.0min
    [Parallel(n_jobs=1)]: Done 26449 tasks       | elapsed:  2.2min
    [Parallel(n_jobs=1)]: Done 28799 tasks       | elapsed:  2.4min
    [Parallel(n_jobs=1)]: Done 31249 tasks       | elapsed:  2.6min
    [Parallel(n_jobs=1)]: Done 33799 tasks       | elapsed:  2.8min
    [Parallel(n_jobs=1)]: Done 36449 tasks       | elapsed:  3.0min
    [Parallel(n_jobs=1)]: Done 39199 tasks       | elapsed:  3.2min
    [Parallel(n_jobs=1)]: Done 42049 tasks       | elapsed:  3.5min
    [Parallel(n_jobs=1)]: Done 44999 tasks       | elapsed:  3.7min
    [Parallel(n_jobs=1)]: Done 48049 tasks       | elapsed:  4.0min
    [Parallel(n_jobs=1)]: Done 51199 tasks       | elapsed:  4.2min
    [Parallel(n_jobs=1)]: Done 54449 tasks       | elapsed:  4.5min
    [Parallel(n_jobs=1)]: Done 57799 tasks       | elapsed:  4.8min
    [Parallel(n_jobs=1)]: Done 61249 tasks       | elapsed:  5.1min
    [Parallel(n_jobs=1)]: Done 64799 tasks       | elapsed:  5.4min
    [Parallel(n_jobs=1)]: Done 68449 tasks       | elapsed:  5.7min
    [Parallel(n_jobs=1)]: Done 72199 tasks       | elapsed:  6.0min
    [Parallel(n_jobs=1)]: Done 76049 tasks       | elapsed:  6.3min
    [Parallel(n_jobs=1)]: Done 79999 tasks       | elapsed:  6.6min
    [Parallel(n_jobs=1)]: Done 84049 tasks       | elapsed:  6.9min
    [Parallel(n_jobs=1)]: Done 88199 tasks       | elapsed:  7.3min
    [Parallel(n_jobs=1)]: Done 92449 tasks       | elapsed:  7.6min
    [Parallel(n_jobs=1)]: Done 96799 tasks       | elapsed:  8.0min
    [Parallel(n_jobs=1)]: Done 101249 tasks       | elapsed:  8.4min
    [Parallel(n_jobs=1)]: Done 105799 tasks       | elapsed:  8.8min
    [Parallel(n_jobs=1)]: Done 110449 tasks       | elapsed:  9.1min
    [Parallel(n_jobs=1)]: Done 115199 tasks       | elapsed:  9.5min
    [Parallel(n_jobs=1)]: Done 120049 tasks       | elapsed:  9.9min
    [Parallel(n_jobs=1)]: Done 124999 tasks       | elapsed: 10.3min
    [Parallel(n_jobs=1)]: Done 130049 tasks       | elapsed: 10.8min
    [Parallel(n_jobs=1)]: Done 135199 tasks       | elapsed: 11.2min
    [Parallel(n_jobs=1)]: Done 140449 tasks       | elapsed: 11.6min
    [Parallel(n_jobs=1)]: Done 145799 tasks       | elapsed: 12.1min
    [Parallel(n_jobs=1)]: Done 151249 tasks       | elapsed: 12.5min
    [Parallel(n_jobs=1)]: Done 156799 tasks       | elapsed: 13.0min
    [Parallel(n_jobs=1)]: Done 162449 tasks       | elapsed: 13.4min
    [Parallel(n_jobs=1)]: Done 168199 tasks       | elapsed: 13.9min
    [Parallel(n_jobs=1)]: Done 174049 tasks       | elapsed: 14.4min
    [Parallel(n_jobs=1)]: Done 179999 tasks       | elapsed: 14.9min
    [Parallel(n_jobs=1)]: Done 186049 tasks       | elapsed: 15.3min
    [Parallel(n_jobs=1)]: Done 192199 tasks       | elapsed: 15.9min
    [Parallel(n_jobs=1)]: Done 198449 tasks       | elapsed: 16.4min
    [Parallel(n_jobs=1)]: Done 204799 tasks       | elapsed: 16.9min
    [Parallel(n_jobs=1)]: Done 211249 tasks       | elapsed: 17.4min
    [Parallel(n_jobs=1)]: Done 217799 tasks       | elapsed: 17.9min
    [Parallel(n_jobs=1)]: Done 224449 tasks       | elapsed: 18.5min
    [Parallel(n_jobs=1)]: Done 231199 tasks       | elapsed: 19.0min
    [Parallel(n_jobs=1)]: Done 238049 tasks       | elapsed: 19.6min
    [Parallel(n_jobs=1)]: Done 244999 tasks       | elapsed: 20.2min
    [Parallel(n_jobs=1)]: Done 252049 tasks       | elapsed: 20.8min
    [Parallel(n_jobs=1)]: Done 259199 tasks       | elapsed: 21.4min
    [Parallel(n_jobs=1)]: Done 266449 tasks       | elapsed: 22.0min
    [Parallel(n_jobs=1)]: Done 273799 tasks       | elapsed: 22.6min
    [Parallel(n_jobs=1)]: Done 281249 tasks       | elapsed: 23.2min
    [Parallel(n_jobs=1)]: Done 288799 tasks       | elapsed: 23.8min
    [Parallel(n_jobs=1)]: Done 296449 tasks       | elapsed: 24.5min
    [Parallel(n_jobs=1)]: Done 304199 tasks       | elapsed: 25.1min
    [Parallel(n_jobs=1)]: Done 312049 tasks       | elapsed: 25.8min
    [Parallel(n_jobs=1)]: Done 319999 tasks       | elapsed: 26.4min
    [Parallel(n_jobs=1)]: Done 328049 tasks       | elapsed: 27.1min
    [Parallel(n_jobs=1)]: Done 336199 tasks       | elapsed: 27.8min
    [Parallel(n_jobs=1)]: Done 344449 tasks       | elapsed: 28.5min
    [Parallel(n_jobs=1)]: Done 352799 tasks       | elapsed: 29.2min
    [Parallel(n_jobs=1)]: Done 361249 tasks       | elapsed: 29.9min
    [Parallel(n_jobs=1)]: Done 369799 tasks       | elapsed: 30.6min
    [Parallel(n_jobs=1)]: Done 378449 tasks       | elapsed: 31.3min
    [Parallel(n_jobs=1)]: Done 387199 tasks       | elapsed: 32.0min
    [Parallel(n_jobs=1)]: Done 396049 tasks       | elapsed: 32.7min
    [Parallel(n_jobs=1)]: Done 404999 tasks       | elapsed: 33.5min
    [Parallel(n_jobs=1)]: Done 414049 tasks       | elapsed: 34.2min
    [Parallel(n_jobs=1)]: Done 423199 tasks       | elapsed: 35.0min
    [Parallel(n_jobs=1)]: Done 432449 tasks       | elapsed: 35.7min
    [Parallel(n_jobs=1)]: Done 441799 tasks       | elapsed: 36.5min
    [Parallel(n_jobs=1)]: Done 451249 tasks       | elapsed: 37.3min
    [Parallel(n_jobs=1)]: Done 460799 tasks       | elapsed: 38.1min
    [Parallel(n_jobs=1)]: Done 470449 tasks       | elapsed: 38.9min
    [Parallel(n_jobs=1)]: Done 480199 tasks       | elapsed: 39.7min
    [Parallel(n_jobs=1)]: Done 490049 tasks       | elapsed: 40.5min
    [Parallel(n_jobs=1)]: Done 499999 tasks       | elapsed: 41.3min
    [Parallel(n_jobs=1)]: Done 510049 tasks       | elapsed: 42.2min
    [Parallel(n_jobs=1)]: Done 520199 tasks       | elapsed: 43.0min
    [Parallel(n_jobs=1)]: Done 530449 tasks       | elapsed: 43.9min
    [Parallel(n_jobs=1)]: Done 540799 tasks       | elapsed: 44.7min
    [Parallel(n_jobs=1)]: Done 551249 tasks       | elapsed: 45.6min
    [Parallel(n_jobs=1)]: Done 561799 tasks       | elapsed: 46.5min
    [Parallel(n_jobs=1)]: Done 572449 tasks       | elapsed: 47.4min
    [Parallel(n_jobs=1)]: Done 583199 tasks       | elapsed: 48.3min
    [Parallel(n_jobs=1)]: Done 594049 tasks       | elapsed: 49.2min
    [Parallel(n_jobs=1)]: Done 604999 tasks       | elapsed: 50.1min
    [Parallel(n_jobs=1)]: Done 616049 tasks       | elapsed: 51.0min
    [Parallel(n_jobs=1)]: Done 627199 tasks       | elapsed: 51.9min
    [Parallel(n_jobs=1)]: Done 638449 tasks       | elapsed: 52.8min
    [Parallel(n_jobs=1)]: Done 649799 tasks       | elapsed: 53.8min
    [Parallel(n_jobs=1)]: Done 661249 tasks       | elapsed: 54.7min
    [Parallel(n_jobs=1)]: Done 672799 tasks       | elapsed: 55.6min
    [Parallel(n_jobs=1)]: Done 684449 tasks       | elapsed: 56.6min
    [Parallel(n_jobs=1)]: Done 696199 tasks       | elapsed: 57.5min
    [Parallel(n_jobs=1)]: Done 708049 tasks       | elapsed: 58.5min
    [Parallel(n_jobs=1)]: Done 719999 tasks       | elapsed: 59.5min
    [Parallel(n_jobs=1)]: Done 732049 tasks       | elapsed: 60.5min
    [Parallel(n_jobs=1)]: Done 744199 tasks       | elapsed: 61.5min
    [Parallel(n_jobs=1)]: Done 756449 tasks       | elapsed: 62.5min
    [Parallel(n_jobs=1)]: Done 768799 tasks       | elapsed: 63.6min
    [Parallel(n_jobs=1)]: Done 781249 tasks       | elapsed: 64.6min
    [Parallel(n_jobs=1)]: Done 793799 tasks       | elapsed: 65.7min
    [Parallel(n_jobs=1)]: Done 806449 tasks       | elapsed: 66.7min
    [Parallel(n_jobs=1)]: Done 819199 tasks       | elapsed: 67.8min
    [Parallel(n_jobs=1)]: Done 832049 tasks       | elapsed: 68.9min
    [Parallel(n_jobs=1)]: Done 844999 tasks       | elapsed: 69.9min
    [Parallel(n_jobs=1)]: Done 858049 tasks       | elapsed: 71.0min
    [Parallel(n_jobs=1)]: Done 871199 tasks       | elapsed: 72.1min
    [Parallel(n_jobs=1)]: Done 884449 tasks       | elapsed: 73.2min
    [Parallel(n_jobs=1)]: Done 897799 tasks       | elapsed: 74.3min
    [Parallel(n_jobs=1)]: Done 911249 tasks       | elapsed: 75.4min
    [Parallel(n_jobs=1)]: Done 924799 tasks       | elapsed: 76.5min
    [Parallel(n_jobs=1)]: Done 938449 tasks       | elapsed: 77.6min
    [Parallel(n_jobs=1)]: Done 952199 tasks       | elapsed: 78.7min
    [Parallel(n_jobs=1)]: Done 966049 tasks       | elapsed: 79.9min
    [Parallel(n_jobs=1)]: Done 979999 tasks       | elapsed: 81.0min
    [Parallel(n_jobs=1)]: Done 994049 tasks       | elapsed: 82.2min
    [Parallel(n_jobs=1)]: Done 1008199 tasks       | elapsed: 83.4min
    [Parallel(n_jobs=1)]: Done 1022449 tasks       | elapsed: 84.5min
    [Parallel(n_jobs=1)]: Done 1036799 tasks       | elapsed: 85.7min
    [Parallel(n_jobs=1)]: Done 1051249 tasks       | elapsed: 86.9min
    [Parallel(n_jobs=1)]: Done 1065799 tasks       | elapsed: 88.1min
    [Parallel(n_jobs=1)]: Done 1080449 tasks       | elapsed: 89.3min
    [Parallel(n_jobs=1)]: Done 1095199 tasks       | elapsed: 90.5min
    [Parallel(n_jobs=1)]: Done 1110049 tasks       | elapsed: 91.7min
    [Parallel(n_jobs=1)]: Done 1124999 tasks       | elapsed: 93.0min
    [Parallel(n_jobs=1)]: Done 1140049 tasks       | elapsed: 94.2min
    [Parallel(n_jobs=1)]: Done 1155199 tasks       | elapsed: 95.4min
    [Parallel(n_jobs=1)]: Done 1170449 tasks       | elapsed: 96.7min
    [Parallel(n_jobs=1)]: Done 1185799 tasks       | elapsed: 97.9min
    [Parallel(n_jobs=1)]: Done 1201249 tasks       | elapsed: 99.2min
    [Parallel(n_jobs=1)]: Done 1216799 tasks       | elapsed: 100.5min
    [Parallel(n_jobs=1)]: Done 1232449 tasks       | elapsed: 101.8min
    [Parallel(n_jobs=1)]: Done 1248199 tasks       | elapsed: 103.1min
    [Parallel(n_jobs=1)]: Done 1264049 tasks       | elapsed: 104.4min
    [Parallel(n_jobs=1)]: Done 1279999 tasks       | elapsed: 105.8min
    [Parallel(n_jobs=1)]: Done 1296049 tasks       | elapsed: 107.1min
    [Parallel(n_jobs=1)]: Done 1312199 tasks       | elapsed: 108.4min
    [Parallel(n_jobs=1)]: Done 1328449 tasks       | elapsed: 109.8min
    [Parallel(n_jobs=1)]: Done 1344799 tasks       | elapsed: 111.1min
    [Parallel(n_jobs=1)]: Done 1361249 tasks       | elapsed: 112.4min
    [Parallel(n_jobs=1)]: Done 1377799 tasks       | elapsed: 113.8min
    [Parallel(n_jobs=1)]: Done 1394449 tasks       | elapsed: 115.1min
    [Parallel(n_jobs=1)]: Done 1411199 tasks       | elapsed: 116.5min
    [Parallel(n_jobs=1)]: Done 1428049 tasks       | elapsed: 117.9min
    [Parallel(n_jobs=1)]: Done 1444999 tasks       | elapsed: 119.3min
    [Parallel(n_jobs=1)]: Done 1462049 tasks       | elapsed: 120.7min
    [Parallel(n_jobs=1)]: Done 1479199 tasks       | elapsed: 122.2min
    [Parallel(n_jobs=1)]: Done 1496449 tasks       | elapsed: 123.6min
    [Parallel(n_jobs=1)]: Done 1513799 tasks       | elapsed: 125.0min
    [Parallel(n_jobs=1)]: Done 1531249 tasks       | elapsed: 126.5min
    [Parallel(n_jobs=1)]: Done 1548799 tasks       | elapsed: 128.0min
    [Parallel(n_jobs=1)]: Done 1566449 tasks       | elapsed: 129.4min
    [Parallel(n_jobs=1)]: Done 1584199 tasks       | elapsed: 130.9min
    [Parallel(n_jobs=1)]: Done 1602049 tasks       | elapsed: 132.3min
    [Parallel(n_jobs=1)]: Done 1619999 tasks       | elapsed: 133.8min
    [Parallel(n_jobs=1)]: Done 1638049 tasks       | elapsed: 135.3min
    [Parallel(n_jobs=1)]: Done 1656199 tasks       | elapsed: 136.8min
    [Parallel(n_jobs=1)]: Done 1674449 tasks       | elapsed: 138.3min
    [Parallel(n_jobs=1)]: Done 1692799 tasks       | elapsed: 139.8min
    [Parallel(n_jobs=1)]: Done 1711249 tasks       | elapsed: 141.3min
    [Parallel(n_jobs=1)]: Done 1729799 tasks       | elapsed: 142.9min
    [Parallel(n_jobs=1)]: Done 1748449 tasks       | elapsed: 144.4min
    [Parallel(n_jobs=1)]: Done 1767199 tasks       | elapsed: 146.0min
    [Parallel(n_jobs=1)]: Done 1786049 tasks       | elapsed: 147.5min
    [Parallel(n_jobs=1)]: Done 1804999 tasks       | elapsed: 149.1min
    [Parallel(n_jobs=1)]: Done 1824049 tasks       | elapsed: 150.7min
    [Parallel(n_jobs=1)]: Done 1843199 tasks       | elapsed: 152.2min
    [Parallel(n_jobs=1)]: Done 1862449 tasks       | elapsed: 153.8min
    [Parallel(n_jobs=1)]: Done 1881799 tasks       | elapsed: 155.4min
    [Parallel(n_jobs=1)]: Done 1901249 tasks       | elapsed: 156.9min
    [Parallel(n_jobs=1)]: Done 1920799 tasks       | elapsed: 158.6min
    [Parallel(n_jobs=1)]: Done 1940449 tasks       | elapsed: 160.2min
    [Parallel(n_jobs=1)]: Done 1960199 tasks       | elapsed: 161.8min
    [Parallel(n_jobs=1)]: Done 1980049 tasks       | elapsed: 163.5min
    [Parallel(n_jobs=1)]: Done 1999999 tasks       | elapsed: 165.2min
    [Parallel(n_jobs=1)]: Done 2020049 tasks       | elapsed: 166.8min
    [Parallel(n_jobs=1)]: Done 2040199 tasks       | elapsed: 168.5min
    [Parallel(n_jobs=1)]: Done 2060449 tasks       | elapsed: 170.2min
    [Parallel(n_jobs=1)]: Done 2080799 tasks       | elapsed: 171.8min
    [Parallel(n_jobs=1)]: Done 2101249 tasks       | elapsed: 173.5min
    [Parallel(n_jobs=1)]: Done 2121799 tasks       | elapsed: 175.2min
    [Parallel(n_jobs=1)]: Done 2142449 tasks       | elapsed: 176.9min
    [Parallel(n_jobs=1)]: Done 2160000 out of 2160000 | elapsed: 195.4min finished


    Fitting 1000 folds for each of 2160 candidates, totalling 2160000 fits
    0.509383333333
    {'dtree__min_samples_leaf': 1, 'dtree__min_samples_split': 3, 'kbest__k': 12, 'dtree__splitter': 'best', 'dtree__max_features': None, 'dtree__max_depth': 3, 'dtree__min_weight_fraction_leaf': 0, 'dtree__class_weight': {0: 0.3, 1: 0.8}, 'dtree__criterion': 'entropy'}
    CPU times: user 3h 15min 23s, sys: 7.57 s, total: 3h 15min 31s
    Wall time: 3h 15min 23s



```python
bi = grid_search.best_estimator_.named_steps['kbest'].get_support()
df = pd.DataFrame(data = list(zip(features_list[1:], bi)), columns=['Feature', 'Selected?'])
selected_features = df[df['Selected?']]['Feature'].tolist()
fi = grid_search.best_estimator_.named_steps['dtree'].feature_importances_ 

df = pd.DataFrame(data = list(zip(selected_features, fi)), columns=['Feature', 'Importance'])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>salary</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bonus</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>long_term_incentive</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>expenses</td>
      <td>0.460502</td>
    </tr>
    <tr>
      <th>4</th>
      <td>deferred_income</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>loan_advances</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>other</td>
      <td>0.132632</td>
    </tr>
    <tr>
      <th>7</th>
      <td>restricted_stock</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>exercised_stock_options</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>shared_receipt_with_poi</td>
      <td>0.124567</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fraction_from_poi</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>fraction_to_poi</td>
      <td>0.282299</td>
    </tr>
  </tbody>
</table>
</div>




```python
clf_fin = pipeline.set_params(**grid_search.best_params_)
test_classifier(clf_fin, my_dataset, features_list)

```

    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=12, score_func=<function f_classif at 0x7f1d1ebf17d0>)), ('dtree', DecisionTreeClassifier(class_weight={0: 0.3, 1: 0.8}, criterion='entropy',
                max_depth=3, max_features=None, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=3,
                min_weight_fraction_leaf=0, presort=False, random_state=13,
                splitter='best'))])
    	Accuracy: 0.86240	Precision: 0.48533	Recall: 0.52950	F1: 0.50646	F2: 0.52004
    	Total predictions: 15000	True positives: 1059	False positives: 1123	False negatives:  941	True negatives: 11877
    



```python
%%time

pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kbest', SelectKBest()),
        ('dtree', DecisionTreeClassifier(random_state=random))])

param_grid = {              
    'kbest__k':[12],
    'dtree__max_features': [None],
    'dtree__criterion': ['entropy'],
    'dtree__max_depth': [3],
    'dtree__min_samples_split': [1],
    'dtree__min_samples_leaf': [1],
    'dtree__min_weight_fraction_leaf': [0],
    'dtree__class_weight': [{1: 0.8, 0: 0.3}, {1: 0.8, 0: 0.35}, {1: 0.8, 0: 0.25}, {1: 0.9, 0: 0.2}, {1: 0.85, 0: 0.15}],
    'dtree__splitter': ['best']
      }

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1', verbose=1) # f1 for binary targets
grid_search.fit(features, labels)
print grid_search.best_score_
print grid_search.best_params_
```

    [Parallel(n_jobs=1)]: Done  49 tasks       | elapsed:    0.3s
    [Parallel(n_jobs=1)]: Done 199 tasks       | elapsed:    1.1s
    [Parallel(n_jobs=1)]: Done 449 tasks       | elapsed:    2.4s
    [Parallel(n_jobs=1)]: Done 799 tasks       | elapsed:    4.2s
    [Parallel(n_jobs=1)]: Done 1249 tasks       | elapsed:    6.5s
    [Parallel(n_jobs=1)]: Done 1799 tasks       | elapsed:    9.4s
    [Parallel(n_jobs=1)]: Done 2449 tasks       | elapsed:   12.8s
    [Parallel(n_jobs=1)]: Done 3199 tasks       | elapsed:   16.7s
    [Parallel(n_jobs=1)]: Done 4049 tasks       | elapsed:   21.0s
    [Parallel(n_jobs=1)]: Done 4999 tasks       | elapsed:   25.9s


    Fitting 1000 folds for each of 5 candidates, totalling 5000 fits
    0.509187445887
    {'dtree__min_samples_leaf': 1, 'dtree__min_samples_split': 1, 'kbest__k': 12, 'dtree__splitter': 'best', 'dtree__max_features': None, 'dtree__max_depth': 3, 'dtree__min_weight_fraction_leaf': 0, 'dtree__class_weight': {0: 0.25, 1: 0.8}, 'dtree__criterion': 'entropy'}
    CPU times: user 25.9 s, sys: 52 ms, total: 26 s
    Wall time: 26 s


    [Parallel(n_jobs=1)]: Done 5000 out of 5000 | elapsed:   26.0s finished



```python
clf_fin = pipeline.set_params(**grid_search.best_params_)
test_classifier(clf_fin, my_dataset, features_list)
```

    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=12, score_func=<function f_classif at 0x7f1d1ebf17d0>)), ('dtree', DecisionTreeClassifier(class_weight={0: 0.25, 1: 0.8}, criterion='entropy',
                max_depth=3, max_features=None, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=1,
                min_weight_fraction_leaf=0, presort=False, random_state=13,
                splitter='best'))])
    	Accuracy: 0.86133	Precision: 0.48203	Recall: 0.53650	F1: 0.50781	F2: 0.52464
    	Total predictions: 15000	True positives: 1073	False positives: 1153	False negatives:  927	True negatives: 11847
    



```python
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf_fin
dump_classifier_and_data(clf, my_dataset, features_list)
```

## 5. Evaluation

### validation strategy
Validation is a way of assessing whether my algorithm is actually doing what I want it to do.
First and foremost spliting dataset into trainning set and testing set is important.
Because it gives estimate of performance on an independent dataset, and serves as check on overfitting. 

Classic mistake is in splitting dataset. 
One wants to have as many data points in the training sets to get the best learning results, and also wants the maximum number of data items in the test set to get the best validation. But there is an inherent trade-off between size of train and test data.
In addition, original dataset might have patterns in the way that the classes are represented are not like big lumps of certian labels.
In this case, we need to split and shuffle the dataset.

I used **Stratified ShuffleSplit**, which keeps the same target distribution in the training and testing datasets. Particularly, it is important in the case of imbalanced datasets like few POI in targets. It reduces the variability in models' predictive performance when using a trian/test split. When using CV for model selection, all folds sould have the same target distribution by using Stratified Kfolds.


### Evaluation metrics

As a result, I got 86% accuracy to classify correctly POI and non-POI based on their insider payments and email statistics. 

In this project, I have a small number of POIs among a large number of employees. Because of this asymmetry, it is important to check both precision and recall. 

Precision is a measure of **exactness**, e.g. How many selected employees are indeed POIs?
    - Precision was 48%, which means out of 100 employees the model predicted as POI, I can be sure that 48 of them are real POIs. 

Recall is a measure of **completeness**, e.g. How many POIs are found?
    - Recall was 53%, which means given total 100 POIs in a dataset, the model can find 53 of them. 




## References

- http://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
- http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
- https://discussions.udacity.com/t/final-project-text-learning/28169/13
- http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
- http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
- https://jaycode.github.io/enron/identifying-fraud-from-enron-email.html
- http://stats.stackexchange.com/questions/62621/recall-and-precision-in-classification


## Appendix


- ### text learning with Enron email
    I accessed all of the emails in the Enron Corpus, I looped through all of the files in the folder 'maildir'. 
    
    I saved word_data, name_data, email_data, poi_data into enron_word_data.pkl. word_data is words in emails of each person, name_data is owner of the emails, email_data is about from/to, poi_data is flag of poi of each person.
    
    There were similar cases in the [forum][qna_text_learning]. However, one coach said that adding features to the dataset baed on the original text corpus goes beyond the expectations of the project. Anyway, I tried predict POI with email text. Accuracy is 0.777778, but precision, recall, f1 are not good. I think more advanced text learning techniques are required to improve the performance.  


```python
from os import listdir
from os.path import isfile
sys.path.append("../tools/")
from parse_out_email_text import parseOutText

word_data = []
name_data = []
email_data = []
poi_data = []
    
email_dataset_file = 'enron_word_data.pkl'

if (isfile(email_dataset_file)):
    email_dataset = pickle.load(open(email_dataset_file, "r"))
    word_data, name_data, email_data, poi_data = zip(*email_dataset)
else:
    def parse_email(email_list, word_data):
        for path in email_list:
            path = '..' + path[19:-1]
            email = open(path, "r")
            text = parseOutText(email)
            word_data.append(text)
            email.close()
        return word_data
    email_columns = ['name', 'email_address', 'poi']
    df_email = df[email_columns]
    for idx, row in df_email.iterrows():
        name = row['name']
        email = row[ 'email_address']
        poi = row['poi']
        print name, email, poi
        if email != 'NaN':
            email_files = [ 'emails_by_address/from_'+email+'.txt', 'emails_by_address/to_'+email+'.txt' ]

            for index, f in enumerate(email_files):
                if isfile(f):
                    email_list = open(f, 'r')
                    word_data = parse_email(email_list, word_data)
                    name_data.append(name)
                    poi_data.append(poi)
                    if index == 0:
                        email_data.append('from')
                    else:
                        email_data.append('to')
                    email_list.close()
    email_dataset = zip(word_data, name_data, email_data, poi_data)
    pickle.dump( email_dataset, open('enron_word_data.pkl', 'w'))
    
# if False:  
#     email_messages = []
#     for idx, message in enumerate(word_data):
#         key = name_data[idx]
#         if email_data[idx] == 'from':
#             if 'from_messages_texts' not in data_dict[key].keys():
#                 data_dict[key]['from_messages_texts'] = []
#             data_dict[key]['from_messages_texts'].append(message)
#         elif email_data[idx] == 'to':
#             if 'to_messages_texts' not in data_dict[key].keys():
#                 data_dict[key]['to_messages_texts'] = []
#             data_dict[key]['to_messages_texts'].append(message)

#     # For people without email messages:
#     for key, value in data_dict.items():
#         if 'from_messages_texts' not in value.keys():
#             data_dict[key]['from_messages_texts'] = []
#         if 'to_messages_texts' not in value.keys():
#             data_dict[key]['to_messages_texts'] = []


#     features_list.append('to_messages_texts')
#     features_list.append('from_messages_texts')
    
# data_dict['SKILLING JEFFREY K']
```


```python
from sklearn import cross_validation
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, poi_data, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test)

selector = SelectPercentile(f_classif, percentile=20)
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train).toarray()
features_test  = selector.transform(features_test).toarray()

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)

i = 0
vocab_list = vectorizer.get_feature_names()
for c in clf.feature_importances_:
    #if c >= .2:
    if c > 0.0 :
        print ('feature importance: ', c, 'index: ', i, 'voca: ', vocab_list[i])
    i += 1
result = pd.DataFrame([[accuracy, precision_score(labels_test, pred), recall_score(labels_test,pred), f1_score(labels_test,pred)]], \
                      columns=['accuracy', 'precision', 'recall', 'f1'])
result
```
