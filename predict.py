import os
import pickle
import hopsworks
import pandas as pd

import requests
from comet_ml import API
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from preprocess import transform_ts_data_into_features_and_target
from paths import DATA_DIR
from logger import get_console_logger

load_dotenv()

comet_api = os.getenv("COMET_ML_API_KEY")
comet_workspace = os.getenv("COMET_ML_WORKSPACE")
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = "naveen"

logger = get_console_logger()

connection = hopsworks.login(project=project_name, api_key_value=api_key)
fs = connection.get_feature_store()

if not Path(DATA_DIR + "/" + 'ohlc_data.parquet').exists():
    logger.info('Fetching data from Feature Store')
    try:
        feature_view = fs.get_feature_view(name="dogecoin", version=1)
    except:
        dogecoin_fg = fs.get_feature_group(name="dogecoin", version=1)
        query = dogecoin_fg.select_all()
        feature_view = fs.create_feature_view(name="dogecoin",
                                      version=1,
                                      description="Read from Dogecoin dataset",
                                      query=query)
    df = feature_view.get_batch_data()
    df.to_parquet(DATA_DIR + "/" + "ohlc_data.parquet", index=False)

def load_production_model_from_registry(
        workspace: str,
        api_key: str,
        model_name: str,
        status: str = 'Production',
) -> Pipeline:
    """Loads the production model from the remote model registry"""

    api = API(api_key)
    model_details = api.get_registry_model_details(workspace, model_name)['versions']
    model_versions = [md['version'] for md in model_details if md['status'] == status]
    if len(model_versions) == 0:
        logger.error('No production model found')
        raise ValueError('No production model found')
    else:
        logger.info(f'Found {status} model versions: {model_versions}')
        model_version = model_versions[0]

    # download model from comet ml registry to local file
    api.download_registry_model(
        workspace,
        registry_name=model_name,
        version=model_version,
        output_path='./Production',
        expand=True
    )

    # load model from local file to memory
    pkl_files = [file for file in os.listdir('./Production') if file.endswith('.pkl')]

    if len(pkl_files) == 0:
        raise FileNotFoundError("No .pkl files found in the Production directory")

    model_file_path = os.path.join('./Production', pkl_files[0])
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    return model

def get_actual_dogecoin_price() -> float:
    """Fetches the current price of Dogecoin from Binance API"""
    response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=DOGEUSDT")
    response.raise_for_status()
    data = response.json()
    return float(data["price"])

latest_features, _ = transform_ts_data_into_features_and_target()
latest_features = latest_features.tail(1)
# Load the production model
model = load_production_model_from_registry(comet_workspace, comet_api, "lightgbm")

predictions = model.predict(latest_features)

actual_price = get_actual_dogecoin_price()
print(f"Actual Dogecoin Price: {actual_price}")

print(f"Predicted Price of Dogecoin: {predictions[0]}")

mae = mean_absolute_error([actual_price], predictions)
print(f"Mean Absolute Error (MAE): {mae}")

timestamp = datetime.now().isoformat()
data = {'timestamp': [timestamp], 'mae': [mae]}
new_data = pd.DataFrame(data)
try:
    connection = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = connection.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
    name='dogecoin_mae',
    version=1,
    primary_key=['timestamp'],
    description='Model performance metrics'
    )
    feature_group.insert(new_data)

except hopsworks.client.exceptions.RestAPIError as e:
    logger.error(f"Error connecting to Hopsworks API: {str(e)}")
    exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    exit(1)