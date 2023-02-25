# Importing required libraries
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import xgboost as xgb

import warnings
# warnings.filterwarnings('ignore')
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

import requests
from bs4 import BeautifulSoup as bs4


resp = requests.get("https://cbr.ru/")
soup = bs4(resp.text, 'lxml')
# print(soup)

qt = soup.find("div", class_="col-md-2 col-xs-9 _right mono-num").text
# qt1 = soup.find("div", class_="col-md-2 col-xs-9 _right mono-num").text
# print(qt)
# print(qt1)
dollar = qt

dollar = dollar[0:6].replace(",", ".")
dollar = float(dollar)

# data_1 = pd.read_csv("data_1.csv")
# rinochniye_result = data_1.copy()

prices_hist = pd.read_csv("prices_hist.csv")
pa = prices_hist.copy()

# pa["price"] = pa["price"].apply(lambda x: x*dollar)
# # print(pa)
# pp = pa.copy()
# print(pp)

# paa = pa[pa['datetime'], pa['price'] * dollar]
# print(paa)
# print(paa)
# price_ruble = paa.copy()
# print(price_ruble)

# filtered = pp.loc[(pp["datetime"] >= "2022-07-01") & (pp["datetime"] <= "2022-12-30")]
# pa = pa.drop(pa.index[pa["datetime"] < "2022-07-01"])
# pa = pa.copy()
# # print(filtered)

# flt_fr = filtered.copy()
# print(flt_fr)
# print(flt_fr)

flt_price = []
for prices in pa["price"]:
    flt_price.append(prices)
    
item = len(flt_price)
# print(item)

math_sum = sum(flt_price)
# print(int(math_sum))

alltime_period = math_sum // item
print(alltime_period)


dt = pd.read_csv("prices_hist.csv") 
# dt.head()



dt = pa.copy()
print(dt)
# dt.info()

min_max_medium = dt.describe() # Печатает статистические данные, такие как среднее, медианное, минимальное, максимальное, стандартное отклонение
print(min_max_medium)

dt['datetime'] = pd.to_datetime(dt['datetime']) # Преобразование времени в объект datetime для дальнейшей обработки
dt.set_index('datetime', inplace=True)
time_fix = dt.head()
# print(b)

plt.figure(figsize=(12, 8)) # Размер графика
dt['price'].plot()
plt.title('История цен')
plt.xlabel('datetime')
plt.ylabel('price')

forecast_out = int(math.ceil(0.0372 * len(dt))) # Прогнозирование 5% данных. Если повысить процент, то увеличится кол-во дней на будущий прогноз
print(forecast_out) # При выводе видно, что мы освободили 14 дней, следовательно дальнейший прогноз будет строиться на 2 недели. 
dt['labels'] = dt['price'].shift(-forecast_out)

scaler = StandardScaler() # Маштабировнаие данных таким образом, чтобы среднее значение из наблюдаемых было равно 0, а стандартное отклонение равнялось -1 (модуль предварительной обработки данных)
X = np.array(dt.drop(columns='labels'))
scaler.fit(X)
X = scaler.transform(X)

X_Predictions = X[-forecast_out:] # Данные, предназначенные для дальнейшего прогноза 
X = X[:-forecast_out] # Данны, предназначенные для обучения модели

dt.dropna(inplace=True) # Получение целевых значений (inplace=True - сохранение изменений в самом DataFrame)
y = np.array(dt['labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Обучение модели с 80% X

# Тестирование нескольких моделей для определения наибольшей точности в прогнозе
lr = LinearRegression() 
# Алгоритм машинного обучения, является оптимизированной версией  простой и множественной линейной регрессии.
# Ищет корреляции между зависимой или одной/несколько независимых переменных.
# Стремится к уменьшению расхождения между фактическими и прогнозируемыми значениями зависимой переменной.
lr.fit(X_train, y_train)
lr_confidence = lr.score(X_test, y_test)
print('Линейная регрессия',lr_confidence)


rf = RandomForestRegressor() 
# Универсальная модель, подходящая не только для регрессионого анализа. Данная модель основана на подвыборке, по которой в дальнейшем строится "Дерево",
# где просматриваются случайные признаки, из которых уже выбирается наилучший признак.
# Дерево строится, как правило, до исчерпания выборки
rf.fit(X_train, y_train)
rf_confidence = rf.score(X_test, y_test)
print('Случайно прогнозируемая регрессия', rf_confidence)

rg = Ridge()
#  Ридж регрессия - модель, предназначенная для достижения наилучшей производительности, в которой ни один коэффициент не должен достичь экстремального значения
rg.fit(X_train, y_train)
rg_confidence = rg.score(X_test, y_test)
print("Ридге", rg_confidence)

svr = SVR() # SVR - Support Vector Regression(Регрессия опорных векторов). 
# Алгоритм работы можно описать довольно простым способом:
# Существует гиперплоскость, на которой распределяются точки(вектора). Основная задача описивыается как: 
# Минимизация ошибки, идентицифируя функцию, которая помещает наибольшее кол-во исходных точек внутри и уменьшая их разброс
svr.fit(X_train, y_train)
svr_confidence = svr.score(X_test, y_test)
print("SVR", svr_confidence)

# Распределение на гистограмме и дальнейшая визуальная демонстрация точности прогноза из 4 моделей
names = ['Linear Regression', 'Random Forest', 'Ridge', 'SVR']
columns = ['model', 'accuracy']
scores = [lr_confidence, rf_confidence, rg_confidence, svr_confidence]
alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns = columns)
print(alg_vs_score)
alg_vs_score = alg_vs_score.copy()


plt.figure(figsize=(8, 6))
sns.barplot(data = alg_vs_score, x='model', y='accuracy')
plt.title('Производительность различных моделей')
plt.xticks(rotation='vertical')


last_date = dt.index[-1] # Получение последних данных в наборе
# print(last_date)
last_unix = last_date.timestamp() # Преобразование времени в секунды
one_day = 86400 # 1 день 86400 - секунд
next_unix = last_unix + one_day # Получение времени в секундах на следующий день
forecast_set = rf.predict(X_Predictions) # Прогнозирование данных
print(forecast_set)
dt['Forecast'] = np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    dt.loc[next_date] = [np.nan for _ in range(len(dt.columns)-1)]+[i]

plt.figure(figsize=(18, 8))
dt['price'].plot()
dt['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
# plt.show()

data1 = pd.read_csv('data_1.csv')
arm = data1.copy()
arm = arm.filter(regex="railway")
arm = arm.dropna(axis=0, how="any")
arm = arm.copy()
# arm.plot(rot=0)
plt.figure(figsize=(20,20))
sns.heatmap(arm.corr(), annot=True)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.show()
# print(dt['price'])

i = 2
ii = 2

while ii >= 2 and ii <= len(forecast_set):
    ii = ii+1
    sr = (forecast_set[i-1] * dollar + forecast_set[i-2] * dollar)/2
    # print(sr)
    # print(forecast_set[i])
    srr = (sr + forecast_set[i] * dollar)/2
    i = i+1
    ag = forecast_set[i] * dollar
    if ag > srr:
        print(f"Прогноз:{ag}, среднее {sr}, ср {srr} Произведите закупку на неделю. Цена должна упасть")
    else:
        print(f"Прогноз:{ag}, среднее {sr}, ср {srr} Произведите закупку на 2 недели. Цена должна повыситься")
