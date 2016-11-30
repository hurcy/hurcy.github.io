---
layout: post
title:  "Investigate a Titanic Dataset"
date:   2016-04-01 23:49:29 +0900
categories: udacity en
---

I investigated Titanic dataset using NumPy and Pandas. I went through the entire data analysis process, starting by posing a question and finishing by sharing my findings. In this report, the passengers survival rate is analyzed according to passenger class, age, and sex.

## Questions

- <a href='#q1'>Does pclass affect to survival rate?</a>
- <a href='#q2'>What age were more likely to survive?</a>
- <a href='#q3'>Does sex affect to survival rate?</a>

## Variables

- independent variables : Pclass, Age, Sex
- dependent variable : Survived

## Reading Titanic data


```python
%pylab inline
import seaborn as sns
import numpy as np
import pandas as pd

titanic_data_df = pd.read_csv('titanic_data.csv')
```

    Populating the interactive namespace from numpy and matplotlib


## Statistics of Titanic data


```python
titanic_data_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print 'Total number of passengers :', titanic_data_df['PassengerId'].count()
print 'Total number of male/female :', titanic_data_df.groupby('Sex').count()['PassengerId']['male'], '/', titanic_data_df.groupby('Sex').count()['PassengerId']['female']
print 'Statistics: '
print titanic_data_df.describe()
```

    Total number of passengers : 891
    Total number of male/female : 577 / 314
    Statistics: 
           PassengerId    Survived      Pclass         Age       SibSp  \
    count   891.000000  891.000000  891.000000  714.000000  891.000000   
    mean    446.000000    0.383838    2.308642   29.699118    0.523008   
    std     257.353842    0.486592    0.836071   14.526497    1.102743   
    min       1.000000    0.000000    1.000000    0.420000    0.000000   
    25%     223.500000    0.000000    2.000000   20.125000    0.000000   
    50%     446.000000    0.000000    3.000000   28.000000    0.000000   
    75%     668.500000    1.000000    3.000000   38.000000    1.000000   
    max     891.000000    1.000000    3.000000   80.000000    8.000000   
    
                Parch        Fare  
    count  891.000000  891.000000  
    mean     0.381594   32.204208  
    std      0.806057   49.693429  
    min      0.000000    0.000000  
    25%      0.000000    7.910400  
    50%      0.000000   14.454200  
    75%      0.000000   31.000000  
    max      6.000000  512.329200  


<a id='q1'></a>
# Does pclass affect to survival rate?


```python
survived_passengers_by_class = titanic_data_df.groupby('Pclass').sum()['Survived']
passengers_by_class = titanic_data_df.groupby('Pclass').count()['PassengerId']

def survived_percentage(passengers, survived):
    return survived / passengers


survived_passengers =  survived_percentage(passengers_by_class, survived_passengers_by_class)

survived_passengers.plot(kind='bar', title='Survival Rate by Pclass')
```

![png](/public/p2-8-1.png)


<a id='q2'></a>
# What age were more likely to survive?


```python
def correlation(x, y):
    std_x = (x - x.mean()) / x.std(ddof=0)
    std_y = (y - y.mean()) / y.std(ddof=0)
    
    return (std_x * std_y).mean()

print 'Pearson\'s r:', correlation(titanic_data_df['Survived'], titanic_data_df['Age'])
print 'Age and Survival have negative correlation. Younger passengers were more likely to survive.'
print ' '

avg_age_by_survived = titanic_data_df.groupby('Survived').mean()['Age']
avg_age_by_survived.plot(kind='bar', title='Average Age of Survivors')
```

    Pearson's r: -0.0779826784139
    Age and Survival have negative correlation. Younger passengers were more likely to survive.
     

![png](/public/p2-10-2.png)



```python
ranged_age_of_survivors = titanic_data_df.groupby(pd.cut(titanic_data_df['Age'], np.arange(0, 90, 10))).mean()
print 'Titanic data with age range (pandas groupby range is refered http://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values)'
ranged_age_of_survivors
```

    Titanic data with age range (pandas groupby range is refered http://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Age</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(0, 10]</th>
      <td>430.843750</td>
      <td>0.593750</td>
      <td>2.640625</td>
      <td>4.268281</td>
      <td>1.843750</td>
      <td>1.421875</td>
      <td>30.434439</td>
    </tr>
    <tr>
      <th>(10, 20]</th>
      <td>447.660870</td>
      <td>0.382609</td>
      <td>2.530435</td>
      <td>17.317391</td>
      <td>0.591304</td>
      <td>0.391304</td>
      <td>29.529531</td>
    </tr>
    <tr>
      <th>(20, 30]</th>
      <td>428.682609</td>
      <td>0.365217</td>
      <td>2.386957</td>
      <td>25.423913</td>
      <td>0.321739</td>
      <td>0.239130</td>
      <td>28.306719</td>
    </tr>
    <tr>
      <th>(30, 40]</th>
      <td>468.690323</td>
      <td>0.445161</td>
      <td>2.090323</td>
      <td>35.051613</td>
      <td>0.374194</td>
      <td>0.393548</td>
      <td>42.496100</td>
    </tr>
    <tr>
      <th>(40, 50]</th>
      <td>483.500000</td>
      <td>0.383721</td>
      <td>1.918605</td>
      <td>45.372093</td>
      <td>0.372093</td>
      <td>0.430233</td>
      <td>41.163181</td>
    </tr>
    <tr>
      <th>(50, 60]</th>
      <td>449.809524</td>
      <td>0.404762</td>
      <td>1.523810</td>
      <td>54.892857</td>
      <td>0.309524</td>
      <td>0.309524</td>
      <td>44.774802</td>
    </tr>
    <tr>
      <th>(60, 70]</th>
      <td>430.882353</td>
      <td>0.235294</td>
      <td>1.529412</td>
      <td>63.882353</td>
      <td>0.176471</td>
      <td>0.352941</td>
      <td>45.910782</td>
    </tr>
    <tr>
      <th>(70, 80]</th>
      <td>438.200000</td>
      <td>0.200000</td>
      <td>1.800000</td>
      <td>73.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>25.936680</td>
    </tr>
  </tbody>
</table>
</div>




```python
ranged_age_of_survivors.plot(kind='line', x='Age', y='Survived', title='Survival Rate by Age Range')
```


![png](/public/p2-12-1.png)


<a id='q3'></a>
## Does sex affect to survival rate?


```python
survivors_by_sex = titanic_data_df.groupby('Sex').sum()['Survived']
print survivors_by_sex
survivors_by_sex.plot(kind='bar', title='Number of Survivors by sex')

```

    Sex
    female    233
    male      109
    Name: Survived, dtype: int64




![png](/public/p2-14-2.png)

