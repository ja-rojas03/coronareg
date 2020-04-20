
# Import libraries to handle data
import csv
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso

# Read data into dataframe using pandas
data = pd.read_csv("data\covid19_italy_region_new.csv", sep = ",")
# Removing columns that only have one value or are redundant
data = data.drop(columns = ['SNo','RegionCode','Country','HospitalizedPatients','IntensiveCarePatients','HomeConfinement','NewPositiveCases'])
# Removed hour in Date since it does not contributes anything
data['Date'] =pd.to_datetime(data['Date']).dt.date

# ## To download datase
# data.to_csv('data.csv')
# files.download('data.csv')

## Creation of artifical variable timedelta. It's the number of day after the first data sample
df_all = data.copy()
fromDate = min(data['Date'])
df_all['timedelta'] = (df_all['Date'] - fromDate).dt.days.astype(int)
print(df_all[['Date', 'timedelta']].tail())
df_all.drop('Date', axis = 1, inplace = True)

#ONEHOT ENCODING BLOCK
Region_Enc = data.select_dtypes(include=['object'])
Region_Enc = pd.get_dummies(Region_Enc['RegionName'], columns='RegionName')
#mergedata = mergedata.drop(['sex','region','smoker'],axis=1)

x = df_all.drop(['RegionName'],axis = 1)
#x = df_all.drop(['RegionName'],axis = 1).values #returns a numpy array
#min_max_scaler = preprocessing.StandardScaler().fit(x)
#x_scaled = min_max_scaler.transform(x)
#df_all_scaled = pd.DataFrame(x_scaled, columns=df_all.drop(['RegionName'],axis = 1).columns)

# df_all_scaled

data = pd.concat([Region_Enc,x],axis=1)


data.columns

# def scatter_plot(feature, target):
#     plt.figure(figsize=(16, 8))
#     plt.scatter(
#         df_all[feature],
#         df_all[target],
#         c='black'
#     )
#     plt.show()

# scatter_plot('timedelta', 'Deaths')

# data.to_csv('encoded_and_normalized_data.csv', index=False)
# files.download('encoded_and_normalized_data.csv')

#df = df_all[df_all['Molise']==1]
#df = pd.concat([df_all[df_all['Abruzzo']==1],df_all[df_all['Molise']==1],df_all[df_all['Lombardia']==1]])

df = data

'''Xs = df.drop(['Latitude', 'Longitude', 'HospitalizedPatients',
                'IntensiveCarePatients', 'TotalHospitalizedPatients', 'HomeConfinement',
                'CurrentPositiveCases', 'NewPositiveCases', 'Deaths'],axis = 1)'''

Xs = df.drop(['Latitude', 'Longitude', 'Deaths','TotalHospitalizedPatients','TestsPerformed','Recovered'],axis = 1)
print("Xs.columns\n", Xs.columns)
scaler_xs = preprocessing.StandardScaler().fit(Xs)
Xs_transformed = scaler_xs.transform(Xs)

y = df['Deaths'].values.reshape(-1,1)
#scaler_y = preprocessing.StandardScaler().fit(y)
#y_transformed = scaler_y.transform(y)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs_transformed, y, scoring='neg_mean_squared_error', cv=40)

mean_MSE = np.mean(MSEs)

print("mean_MSE \n",mean_MSE*-1)
print("mean_MSE sqrt \n",np.sqrt(mean_MSE*-1))

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
lin_reg = LinearRegression().fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
lin_reg.score(X_test_transformed, y_test)

np.concatenate((lin_reg.predict(X_test_transformed), y_test), axis=1)

## To download linear regression results
a = np.concatenate((lin_reg.predict(X_test_transformed), y_test), axis=1)
y_predict = lin_reg.predict(X_test_transformed)
"""
To this point "y_predict" works great , problem is on funct newCasePred()
"""
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n", y_predict)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n", X_test_transformed)
# np.savetxt("lin_reg.csv", a, delimiter=",")
# files.download('lin_reg.csv')

## RIDGE REGRESION
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs_transformed, y)

ridge_regressor.best_params_

print("ridge_regressor.best_score_\n", ridge_regressor.best_score_*-1)
print("ridge_regressor.best_score_ sqrt \n",np.sqrt(ridge_regressor.best_score_*-1))

## RIDGE REGRESION To train and download
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
ridge = Ridge(alpha=0).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
ridge.score(X_test_transformed, y_test)
a = np.concatenate((ridge.predict(X_test_transformed), y_test), axis=1)
# np.savetxt("ridge.csv", a, delimiter=",")
# files.download('ridge.csv')

# ## Lasso REGRESION
# lasso = Lasso(max_iter = 100000)

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

# lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 20)

# lasso_regressor.fit(Xs_transformed, y)

# lasso_regressor.best_params_

# lasso_regressor.best_score_
# print("lasso_regressor.best_score_\n", lasso_regressor.best_score_*-1)
# print("lasso_regressor.best_score_ sqrt\n", np.sqrt(lasso_regressor.best_score_*-1))

def convertToCsv(info):
    date_format = "%Y-%m-%d"
    firstDate = "2020-02-24"
    a = datetime.datetime.strptime(firstDate, date_format)
    b = datetime.datetime.strptime(info['date'], date_format)
    deltaTime = b - a
    deltaTime = deltaTime.days

    regions = {
        "Abruzzo": ["1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Basilicata": ["0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Calambria": ["0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Campania": ["0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Emilia-Romagna": ["0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Friuli Venezia Giulia": ["0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Lazio": ["0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Liguria": ["0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Lombardia": ["0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Marche": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Molise": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "P.A Bolzano": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        "P.A Trento": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0"],
        "Piemonte": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0"],
        "Puglia": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0"],
        "Sardegna": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0"],
        "Sicilia": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0"],
        "Toscana": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0"],
        "Umbria": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"],
        "Valle d' Aosta": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0"],
        "Veneto": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1"],
    }

    csv_value = regions[info['selectedRegion']] + \
        [str(info['currentPositive'])] + \
        [str(info['totalPositive'])] + \
        [str(deltaTime)]
    
    atributes = ["Abruzzo","Basilicata","Calabria","Campania","Emilia-Romagna","Friuli Venezia Giulia","Lazio","Liguria","Lombardia","Marche","Molise","P.A. Bolzano","P.A. Trento","Piemonte","Puglia","Sardegna","Sicilia","Toscana","Umbria","Valle d'Aosta","Veneto","CurrentPositiveCases","TotalPositiveCases","timedelta"]
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",csv_value)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",atributes)
    f = open("data/case.csv", "w+")
    writer = csv.writer(f)
    writer.writerows([atributes,csv_value])
    f.close()
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",csv_value)

    #after convert
    convertedData = pd.read_csv("data/case.csv", sep=",")
    return newCasePred(convertedData);

def reTrain() :
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.4, random_state=0)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    lin_reg = LinearRegression().fit(X_train_transformed, y_train)
    X_test_transformed = scaler.transform(X_test)
    lin_reg.score(X_test_transformed, y_test)
    
def newCasePred(Xs_new):
    # Xs_new = pd.read_csv("data\case.csv", sep=",")
    # reTrain()
    scaler_new_xs = preprocessing.StandardScaler().fit(Xs_new)
    new_Xs_transformed = scaler_new_xs.transform(Xs_new)

    prediction = lin_reg.predict(new_Xs_transformed)
    return prediction
    """
    prediction always returning same value 
    """



# info = {
#     "selectedRegion": "Veneto",
#     "totalPositive": "6140",
#     "currentPositive": "1343",
#     "date": "2020-04-03",
# }
# convertToCsv(info)
 
