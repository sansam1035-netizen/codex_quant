import datetime

def time_regime():
    h = datetime.datetime.utcnow().hour
    if 0 <= h < 6:
        return "ASIA"
    if 6 <= h < 13:
        return "EU"
    if 13 <= h < 21:
        return "US"
    return "OFF"
