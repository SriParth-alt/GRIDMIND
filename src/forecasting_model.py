import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def train_models(data):

    features = [
        'hour',
        'day_of_week',
        'is_weekend',
        'is_peak_hour',
        'occupancy_ratio',
        'rolling_load_24h',
        'efficiency_ratio',
        'air_temperature',
        'cloud_coverage',
        'dew_temperature',
        'wind_speed'
    ]

    data = data.dropna(subset=features)

    X = data[features]
    y = data['meter_reading']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    results = {
        "Linear Regression": evaluate(y_test, lr_preds),
        "Random Forest": evaluate(y_test, rf_preds),
        "XGBoost": evaluate(y_test, xgb_preds)
    }

    return results, xgb_model