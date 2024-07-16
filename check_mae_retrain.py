import os
import hopsworks
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
        latest_mae = df['mae'].iloc[-1]
        return latest_mae
    except Exception as e:
        print(f"Error fetching MAE: {str(e)}")
        return None

if __name__ == "__main__":
    mae = fetch_mae_from_hopsworks()
    MAE_THRESHOLD=0.05
    if mae is not None:
        print(f"Latest MAE: {mae}")
        if mae > MAE_THRESHOLD:
            print("MAE exceeds threshold. Triggering model retraining...")
    else:
        print("Failed to fetch MAE from Hopsworks.")