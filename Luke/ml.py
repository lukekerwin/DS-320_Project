import pandas as pd

class ContractPredictor:
    def __init__(self, contract_data, statistics_data):
        self.contract_data = contract_data
        self.statistics_data = statistics_data
        self.dataset = self.__merge_data()

    def __merge_data(self):
        contract_data_ = pd.DataFrame(self.contract_data)
        stats_data_ = pd.DataFrame(self.statistics_data)
        dataset = []
        for season in sorted(contract_data_['DATE'].unique()):
            print(f'--- {season} ---')
            contract_data = contract_data_[contract_data_['DATE'].astype(int)==season]
            contract_data = contract_data[contract_data['STRUCTURE'] == '1-way']
            contract_data = contract_data[contract_data['EXTENSION'] == 0]
            stats_data = stats_data_[stats_data_['SEASON'].astype(int).isin([season-3, season-2, season-1])]

            stats_data = stats_data.groupby('PLAYER').agg({'A':'sum', 'BLK':'sum', 'EVA':'sum', 'EVG':'sum', 'EVSH':'sum', 'FOL':'sum', 'FOW':'sum',
                                                        'G':'sum', 'GP':'sum', 'GWG':'sum', 'HIT':'sum', 'PIM':'sum', 'PLUSMINUS':'sum', 'PPA':'sum',
                                                        'PPG':'sum', 'PPSH':'sum', 'PS':'mean', 'PTS':'sum', 'S':'sum', 'TOI':'sum'}).reset_index()
            for col in ['A', 'BLK', 'EVA', 'EVG', 'EVSH', 'FOL', 'FOW', 'G', 'GWG', 'HIT', 'PIM', 'PLUSMINUS', 'PPA', 'PPG', 'PPSH', 'PTS', 'S', 'TOI']:
                stats_data[col] = round(stats_data[col]/stats_data['GP'],3)

            data = pd.merge(contract_data, stats_data, on='PLAYER', how='left').dropna()
            data.drop(columns=['id', 'TEAM', 'DATE', 'EXTENSION', 'STRUCTURE', 'TYPE'], inplace=True)
            def get_pos(pos):
                # if C and any of LW or RW, then F
                if 'C' in pos and ('LW' in pos or 'RW' in pos):
                    return 'F'
                # if LW and RW, then W
                elif 'LW' in pos or 'RW' in pos:
                    return 'W'
                elif 'C' in pos:
                    return 'C'
                elif 'D' in pos:
                    return 'D'
                else:
                    return 'G'
            data['POS'] = data['POS'].apply(get_pos)
            data.drop(columns=['PLAYER','LENGTH','VALUE'], inplace=True)
            dataset.append(data)
        dataset = pd.concat(dataset)
        return dataset
    
    def predict(self):
        pass

import requests

contract_data = requests.get('http://127.0.0.1:5000/api/contracts').json() # Comment out Brian and use lines below
stats_data = requests.get('http://127.0.0.1:5000/api/statistics').json() # Comment out Brian and use lines below
# contract_data = pd.read_csv('Luke/data/contracts.csv')
# stats_data = pd.read_csv('Luke/data/statistics.csv')

contract_data = pd.DataFrame(contract_data)
stats_data = pd.DataFrame(stats_data)

cp = ContractPredictor(contract_data, stats_data)

data = cp.dataset

data = pd.get_dummies(data, columns=['POS'])

# Train Test Split

from sklearn.model_selection import train_test_split

X = data.drop(columns=['CAP_HIT'])
y = data['CAP_HIT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=716)

# Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find the optimal number of features

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=5)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)


# Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_selected, y_train)

print(f'Linear Regression Train Score: {lr.score(X_train_selected, y_train)}')
print(f'Linear Regression Test Score: {lr.score(X_test_selected, y_test)}')

# Ridge Regression

from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(X_train_selected, y_train)

print(f'Ridge Regression Train Score: {ridge.score(X_train_selected, y_train)}')
print(f'Ridge Regression Test Score: {ridge.score(X_test_selected, y_test)}')

# Lasso Regression

from sklearn.linear_model import Lasso

lasso = Lasso()

lasso.fit(X_train_selected, y_train)

print(f'Lasso Regression Train Score: {lasso.score(X_train_selected, y_train)}')
print(f'Lasso Regression Test Score: {lasso.score(X_test_selected, y_test)}')

# ElasticNet Regression

from sklearn.linear_model import ElasticNet

en = ElasticNet()

en.fit(X_train_selected, y_train)

print(f'ElasticNet Regression Train Score: {en.score(X_train_selected, y_train)}')
print(f'ElasticNet Regression Test Score: {en.score(X_test_selected, y_test)}')

# Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_train_selected, y_train)

print(f'Random Forest Regression Train Score: {rfr.score(X_train_selected, y_train)}')
print(f'Random Forest Regression Test Score: {rfr.score(X_test_selected, y_test)}')

# Gradient Boosting Regression

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

gbr.fit(X_train_selected, y_train)

print(f'Gradient Boosting Regression Train Score: {gbr.score(X_train_selected, y_train)}')
print(f'Gradient Boosting Regression Test Score: {gbr.score(X_test_selected, y_test)}')

# XGBoost Regression

from xgboost import XGBRegressor

xgbr = XGBRegressor()

xgbr.fit(X_train_selected, y_train)

print(f'XGBoost Regression Train Score: {xgbr.score(X_train_selected, y_train)}')
print(f'XGBoost Regression Test Score: {xgbr.score(X_test_selected, y_test)}')

# Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X_train_selected, y_train, epochs=100, batch_size=32, verbose=0)

print(f'Neural Network Train Score: {model.evaluate(X_train_selected, y_train, verbose=0)[1]}')
print(f'Neural Network Test Score: {model.evaluate(X_test_selected, y_test, verbose=0)[1]}')

# Neural Network with Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X_train_selected, y_train, epochs=100, batch_size=32, verbose=0)

print(f'Neural Network with Dropout Train Score: {model.evaluate(X_train_selected, y_train, verbose=0)[1]}')
print(f'Neural Network with Dropout Test Score: {model.evaluate(X_test_selected, y_test, verbose=0)[1]}')

# Voting Regressor

from sklearn.ensemble import VotingRegressor

vr = VotingRegressor([('lr', lr), ('ridge', ridge), ('lasso', lasso), ('en', en), ('rfr', rfr), ('gbr', gbr), ('xgbr', xgbr)])

vr.fit(X_train_selected, y_train)

print(f'Voting Regressor Train Score: {vr.score(X_train_selected, y_train)}')
print(f'Voting Regressor Test Score: {vr.score(X_test_selected, y_test)}')