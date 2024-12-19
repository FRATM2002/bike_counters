import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
import holidays

# Define Useful function
def df_manipulation(df):
    df['date'] = pd.to_datetime(df['date'])
    # extract useful information from dates
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df['weekend_day'] = np.where(df['weekday'].isin([5, 6]), 1, 0)

    season_mapping = {
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn',
        12: 'winter', 1: 'winter', 2: 'winter'
    }

    # Map the 'month' column to seasons
    df['season'] = df['month'].map(season_mapping)

    # consider holidays in France
    holid = holidays.France(years=df['year'].unique())
    df['holidays'] = np.where(df['date'].isin(holid), 1, 0)

    #consider lockdowns
    lockdown_1_start = pd.Timestamp('2020-10-30')
    lockdown_1_end = pd.Timestamp('2020-12-15')

    lockdown_2_start = pd.Timestamp('2021-04-03')
    lockdown_2_end = pd.Timestamp('2021-05-03')

    # Create a lockdown flag using conditions
    df['lockdown'] = 0  # Initialize column with 0
    df.loc[(df['date'] >= lockdown_1_start) & (df['date'] <= lockdown_1_end), 'lockdown'] = 1
    df.loc[(df['date'] >= lockdown_2_start) & (df['date'] <= lockdown_2_end), 'lockdown'] = 1

    return df

df = pd.read_parquet('data/train.parquet')
df = df_manipulation(df)

# create X and y
y = df['log_bike_count']
X = df.drop(columns=['bike_count', 'log_bike_count', 'counter_id',
                            'site_id', 'coordinates', 'counter_technical_id',
                            'site_name', 'date'])

X

# Divide in train and test
df_train, df_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model
variables = ['counter_name', 'year', 'month', 'weekend_day', 'weekday', 'hour', 
             'counter_installation_date', 'season']

preprocess = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown="ignore"), variables)
    ],
    remainder='passthrough'
)

xgb = XGBRegressor(random_state=42)

pipe = Pipeline(steps=[
                ('preprocess', preprocess),
                ('regressor', xgb)
])

param_grid = {
    'regressor__max_depth': [8, 12,15],
    'regressor__n_estimators': [600, 700, 800],
    'regressor__learning_rate': [0.1, 0.01]
}

# Perform grid search
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search_results = grid_search.fit(df_train, y_train)
print("The best parameters are ",grid_search.best_params_)

# Fit the model
xgb = grid_search.best_estimator_
xgb.fit(df_train, y_train)

y_test_pred = xgb.predict(df_test)
rmse = MSE(y_test, y_test_pred)
print(f"The RMSE is: {np.sqrt(rmse)}")

# Test
X_test = pd.read_parquet('data/final_test.parquet')
X_test['date'] = pd.to_datetime(X_test['date'])
X_test = df_manipulation(X_test)

X_test = df_manipulation(X_test)
y_pred = xgb.predict(X_test)

# CSV file
y_pred = xgb.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)