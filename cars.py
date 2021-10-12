import os
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split


#: 1. Dataset manipulations
# There are different column names in files, therefore we are creating dictionary for all unique columns names
columns = {'model': 'model',
           'year': 'year',
           'price': 'price',
           'transmission': 'transmission',
           'mileage': ['mileage', 'mileage2'],
           'fuel_type': ['fuel type', 'fuelType', 'fuel type 2'],
           'tax': ['tax', 'tax(£)'],
           'mpg': 'mpg',
           'engine_size': ['engine size', 'engine size2', 'engineSize']}

# Combining all files into DataFrame
path = 'Datasets/uk_used_cars'
df = pd.DataFrame()
for file in os.listdir(path):
    if file.find('unclean') != -1: # do not include unclean files with wrong data
        pass
    else:
        temp = pd.read_csv(os.path.join(path, file), names=columns, header=None, index_col=False, skiprows=1)
        temp['brand'] = file.split(".")[0]
        df = df.append(temp)

# Checking for null values in the Dataframe
plt.subplots()
msno.bar(df, figsize=(12, 8))
plt.title('Пустые значения по столбцам')

brands_with_na = df.loc[df['mpg'].isna()]
brands_with_na = list(brands_with_na['brand'].unique())

# Set categorical type
categorical = ['transmission', 'fuel_type', 'brand', 'model']
df[categorical] = df[categorical].astype('category')

# v_1 with matplotlib
by_fuel_type = df.groupby('fuel_type').agg({'fuel_type': 'count'}).rename(
    columns={'fuel_type': 'values'}).reset_index().sort_values(by='values', ascending=False)
plt.subplots()
plt.bar(by_fuel_type['fuel_type'], by_fuel_type['values'])
plt.title('Количество машин по типу двигателя')

# Replace cclass and focus with name of the brand
to_replace = {'cclass': 'merc',
              'focus': 'ford'}
df['brand'] = df.brand.map(to_replace).fillna(df.brand)

# v_2 with seaborn
plt.subplots()
sns.countplot(x='brand',
              data=df,
              order=df['brand'].value_counts().index,
              palette='mako')
plt.title('Количество машин по брендам')

# Add column with period of use
df['years_of_use'] = 2018 - df['year']
conditions = [df['years_of_use'] < 3, df['years_of_use'] < 6, df['years_of_use'] < 10, df['years_of_use'] >= 10]
outputs = ['less than 3', 'from 3 to 6', 'from 6 to 10', 'more than 10']
df['period_of_use'] = np.select(conditions, outputs)
df['period_of_use'] = df['period_of_use'].astype('category')

# Pivot table
average_price = df.pivot_table(index='brand',
                            columns='period_of_use',
                            values='price',
                            fill_value=0, )

# Distribution of prices by brands
grouped = df.groupby(['period_of_use', 'brand'])['price'].mean().reset_index()

plt.subplots()
sns.boxplot(x='brand', y='price', data=grouped)
plt.title('Average price by brands')

# TODO: изменения цен с каждым годом пробега
# TODO: почему выводится пустой график

#: 2. Linear regression on BMW
# корреляционная матрица
bmw = df[df['brand']=='bmw']
cor_mat = bmw.corr()
plt.subplots()
sns.heatmap(cor_mat, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
plt.title('Корреляционная матрица')

# гистрограмма по цене
plt.subplots()
sns.histplot(bmw['price'], color='black')
plt.title('Гистограмма распределения цен')
descriptive_stat = df.describe()

# Train/test split
# TODO: сделать деление на тренировочную и тестовую выборку с sklearn.train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

num_training = int(0.8 * bmw['price'].count())
num_test = bmw['price'].count() - num_training

# Training data
X_train = np.array(bmw['mileage'][:num_training]).reshape((num_training, 1))
y_train = np.array(bmw['price'][:num_training])

# Test data
X_test = np.array(bmw['year'][num_training:]).reshape((num_test, 1))
y_test = np.array(bmw['price'][num_training:])

# Create linear regression object
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)

plt.subplots()
plt.scatter(X_test, y_test, color='red', marker='.', linewidths=1)
plt.plot(X_test, y_test_pred, color='black', linewidth=1)
plt.xticks(())
plt.yticks(())

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
