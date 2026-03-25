import numpy as np


def engineer_features(data):

    # Time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Peak hour flag (simulate grid stress hours)
    data['is_peak_hour'] = data['hour'].apply(lambda x: 1 if 12 <= x <= 18 else 0)

    # Simulated occupancy (logic-based, not random)
    def simulate_occupancy(row):
        if row['is_weekend'] == 1:
            return 0.3
        elif 8 <= row['hour'] <= 17:
            return 0.9
        else:
            return 0.5

    data['occupancy_ratio'] = data.apply(simulate_occupancy, axis=1)

    # Rolling load average per building (microgrid intelligence)
    data['rolling_load_24h'] = (
        data.groupby('building_id')['meter_reading']
        .transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    )

    # Efficiency ratio
    data['efficiency_ratio'] = data['meter_reading'] / (data['occupancy_ratio'] + 0.1)

    return data