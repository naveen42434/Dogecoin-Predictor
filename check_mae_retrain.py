import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = "naveen"

connection = hopsworks.login(project=project_name, api_key_value=api_key)
fs = connection.get_feature_store()


def fetch_mae_from_hopsworks():
    try:
        fs = connection.get_feature_store()
        feature_group = fs.get_feature_group(name='dogecoin_mae', version=1)
        df = feature_group.read()

        all_maes = df['mae'].tolist()
        return all_maes
    except Exception as e:
        print(f"Error fetching MAE: {str(e)}")
        return None


def should_retrain(mae_values, window_size=5, threshold=0.05):
    if len(mae_values) < window_size:
        return False

    rolling_avg = pd.Series(mae_values).rolling(window=window_size).mean().tolist()
    print(f"Rolling window MAE: {rolling_avg[-1]}")

    if rolling_avg[-1] > threshold:
        return True

    return False


if __name__ == "__main__":
    mae_values = fetch_mae_from_hopsworks()
    if mae_values is not None:
        retrain = should_retrain(mae_values)
        if retrain:
            print("MAE exceeds threshold. Triggering model retraining...")
        else:
            print("MAE within acceptable range. No retraining needed.")
    else:
        print("Failed to fetch MAE from Hopsworks.")