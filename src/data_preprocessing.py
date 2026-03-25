import pandas as pd


def preprocess_data(train, building_metadata, weather):
    # Convert timestamp to datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])

    # Merge building metadata
    data = train.merge(building_metadata, on='building_id', how='left')

    # Merge weather data
    data = data.merge(weather, on=['site_id', 'timestamp'], how='left')

    # Drop rows with missing meter_reading
    data = data.dropna(subset=['meter_reading'])

    # Fill missing weather values
    data = data.ffill()
    return data