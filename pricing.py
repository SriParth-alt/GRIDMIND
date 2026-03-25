def get_price(hour):
    if 18 <= hour <= 22:
        return 10  # peak
    elif 10 <= hour <= 17:
        return 6   # mid
    else:
        return 3   # off-peak