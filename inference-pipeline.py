import os
import pickle
import requests
import numpy as np
import streamlit as st
import hopsworks
import pandas as pd
from comet_ml import API
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Union
from pathlib import Path

load_dotenv()

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
    COMET_ML_API_KEY  = os.environ['COMET_ML_API_KEY']
    COMET_ML_WORKSPACE = os.environ['COMET_ML_WORKSPACE']
except:
    raise Exception('Set environment variable')

######################## Helper functions ########################

def fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)

def transform_ts_data_into_features_and_target(
        # ts_data: pd.DataFrame,
        path_to_input: Optional[Path] =  'ohlc_data.parquet',
        input_seq_len: Optional[int] = 24,
        step_size: Optional[int] = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    # load parquet file
    ts_data = pd.read_parquet(path_to_input)
    ts_data = ts_data[['time', 'close']]
    ts_data.sort_values(by=['time'], inplace=True)

    # pre-compute cutoff indices to split dataframe rows
    indices = get_cutoff_indices_features_and_target(
        ts_data,
        input_seq_len,
        step_size
    )

    # slice and transpose data into numpy arrays for features and targets
    n_examples = len(indices)
    x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
    y = np.ndarray(shape=(n_examples), dtype=np.float32)
    times = []
    for i, idx in enumerate(indices):
        x[i, :] = ts_data.iloc[idx[0]:idx[1]]['close'].values
        y[i] = ts_data.iloc[idx[1]:idx[2]]['close'].values
        times.append(ts_data.iloc[idx[1]]['time'])

    # numpy -> pandas
    features = pd.DataFrame(
        x,
        columns=[f'price_{i + 1}_hour_ago' for i in reversed(range(input_seq_len))]
    )

    # add back column with the time
    # features['time'] = times

    # numpy -> pandas
    targets = pd.DataFrame(y, columns=[f'target_price_next_hour'])

    return features, targets['target_price_next_hour']


def get_cutoff_indices_features_and_target(
        data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
) -> List[Tuple[int, int, int]]:
    stop_position = len(data) - 1

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices


def process_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for prediction.
    """
    latest_features, target = transform_ts_data_into_features_and_target()
    # retrieve the latest feature
    df = latest_features.tail(1)
    actual_target = target.tail(1).values

    return df,actual_target

def get_actual_dogecoin_price() -> float:
    """Fetches the current price of Dogecoin from Binance API"""
    try:
        response = requests.get("https://api.pro.coinbase.com/products/DOGE-USD/ticker")
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dogecoin price: {str(e)}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"Error parsing Dogecoin price data: {str(e)}")
        return None

def get_model(
        workspace: str,
        api_key: str,
        model_name: str,
        status: str = 'Production',
) -> Pipeline:
    """Retrieve desired model from the Cometml Model Registry."""

    api = API(api_key)
    model_details = api.get_registry_model_details(workspace, model_name)['versions']
    model_versions = [md['version'] for md in model_details if md['status'] == status]
    if len(model_versions) == 0:
        st.error('No production model found')
        raise ValueError('No production model found')
    else:
        # st.info(f'Found {status} model versions: {model_versions}')
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


############################### Streamlit app ##############################

st.title('DOGECOIN Price Prediction Project')
st.write(36 * "-")
st.write("This app uses a machine learning model to predict the hourly price of DOGECOIN.")
st.write("")
st.write(
    "This Streamlit app demonstrates on-demand retrieval of data from the Hopsworks Feature Store, loading a model from the CometML Model Registry, and making predictions.")

st.write(" - For the historical data, the current model has achieved a mean absolute error (MAE) of 0.00133.")
st.write(" - This model is designed to predict hourly price changes of Dogecoin with high accuracy.")
st.write(" - A baseline model that predicts the price will remain the same as the previous hour would typically have a higher MAE.")
st.write("")
st.write("Note: The model performance and predictions can vary based on the market conditions and the quality of the input data.")

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.write(36 * "-")
fancy_header('\n📡 Connecting to Hopsworks Feature Store...')

########### Connect to Hopsworks Feature Store and get Feature Group

project = hopsworks.login(project="naveen",api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

dogecoin_fg = fs.get_feature_group(
    name="dogecoin",
    version=1
)

st.write("Successfully connected!✔️")
progress_bar.progress(20)

########### Get data from Feature Store

st.write(36 * "-")
fancy_header('\n☁️ Retrieving data from Feature Store...')

try:
    feature_view = fs.get_feature_view(name="dogecoin", version=1)
except:
        dogecoin_fg = fs.get_feature_group(name="dogecoin", version=1)
        ds_query = dogecoin_fg.select_all()
        feature_view = fs.create_feature_view(name="dogecoin",
                                      version=1,
                                      description="Read from Dogecoin dataset",
                                      query=ds_query)

data = feature_view.get_batch_data()
data.to_parquet("ohlc_data.parquet", index=False)
data = pd.read_parquet(r"ohlc_data.parquet")

st.write("Successfully retrieved Data!✔️")
progress_bar.progress(40)

########### Prepare data for prediction

st.write(36 * "-")
fancy_header('\n☁️ Processing Data for prediction...')

df,target = process_for_prediction(data)

st.write("Successfully processed data!✔️")

progress_bar.progress(60)

###########  Load model from Hopsworks Model Registry

st.write(36 * "-")
fancy_header(f"Loading Best Model...")

model = get_model(COMET_ML_WORKSPACE,
                  COMET_ML_API_KEY,
                  model_name="lightgbm")

st.write("Successfully loaded!✔️")
progress_bar.progress(70)

########### Predict Price for next hour

st.write(36 * "-")
fancy_header(f"Predicting Price for next hour...")

predictions = model.predict(df)

st.markdown(f"<span style='color: green; font-size: 20px;'>Predicted Price for the next hour:</span> <span style='color: blue; font-size: 20px;'>{predictions[0]}</span>", unsafe_allow_html=True)

st.write(36 * "-")
fancy_header(f"Actual Price for next hour...")

actual_price = get_actual_dogecoin_price()
st.markdown(f"<span style='color: green; font-size: 20px;'>Actual Dogecoin Price for the next hour:</span> <span style='color: blue; font-size: 20px;'>{actual_price}</span>", unsafe_allow_html=True)

progress_bar.progress(85)

st.write(36 * "-")
fancy_header(f"Evaluating Metrics...")

mae = mean_absolute_error([actual_price], predictions)

# Show accuracy
st.markdown(f"**Mean Absolute Error (MAE):** {mae}")

progress_bar.progress(100)

st.button("Re-run")