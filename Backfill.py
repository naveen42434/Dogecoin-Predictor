import os
import pandas as pd
import requests
import hopsworks
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

from paths import DATA_DIR
from logger import get_console_logger

BACKFILL=False
load_dotenv()

logger = get_console_logger(name='dataset_generation')
project_name = "naveen"
api_key = os.getenv("HOPSWORKS_API_KEY")

def download_ohlc_data_from_coinbase(
        product_id: Optional[str] = "DOGE-USD",
        from_day: Optional[str] = "2024-06-20",
        to_day: Optional[str] = "2024-06-30",
) -> Path:
    """
    Downloads historical data from coinbase API and saves data to disk
    Reference: https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles/
    """
    # create list of days as strings
    days = pd.date_range(start=from_day, end=to_day, freq="1D")
    days = [day.strftime("%Y-%m-%d") for day in days]

    # create empty dataframe
    data = pd.DataFrame()

    # create download dir folder if it doesn't exist
    if not Path(DATA_DIR + "/" + 'downloads').exists():
        logger.info('Create directory for downloads')
        os.mkdir(DATA_DIR + "/" + 'downloads')

    for day in days:

        # download file if it doesn't exist
        file_name = DATA_DIR + "/" + 'downloads' + "/" + f'{day}.parquet'
        if Path(file_name).exists():
            logger.info(f'File {file_name} already exists, skipping')
            data_one_day = pd.read_parquet(file_name)
        else:
            logger.info(f'Downloading data for {day}')
            data_one_day = download_data_for_one_day(product_id, day)
            data_one_day.to_parquet(file_name, index=False)

        # combine today's file with the rest of the data
        data = pd.concat([data, data_one_day])

    # save data to disk
    # data.to_parquet(DATA_DIR / f"ohlc_from_{from_day}_to_{to_day}.parquet", index=False)
    data = data.drop_duplicates(subset='time')

    data.to_parquet(DATA_DIR + "/" + f"ohlc_data.parquet", index=False)

    return DATA_DIR + "/" + f"ohlc_data.parquet"

def download_data_for_one_day(product_id: str, day: str) -> pd.DataFrame:
    """
    Downloads one day of data and returns pandas Dataframe
    """
    from datetime import datetime, timedelta
    start = f'{day}T00:00:00'
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    end = f'{end}T00:00:00'

    # call API
    URL = f'https://api.exchange.coinbase.com/products/{product_id}/candles?start={start}&end={end}&granularity=3600'
    r = requests.get(URL)
    data = r.json()

    # transform list of lists to pandas dataframe and return
    return pd.DataFrame(data,columns=['time', 'low', 'high', 'open', 'close', 'volume'])


def download_last_hour_data(product_id: str) -> pd.DataFrame:
    """
    Downloads data for the previous hour from Coinbase API and returns pandas DataFrame.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(hours=1)
        end_str = end.strftime('%Y-%m-%dT%H:%M:%S')
        start_str = start.strftime('%Y-%m-%dT%H:%M:%S')

        # construct URL
        URL = f'https://api.exchange.coinbase.com/products/{product_id}/candles?start={start}&end={end}&granularity=3600'
        # print(f"Fetching data from URL: {URL}")

        # call API
        r = requests.get(URL)
        r.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        data = r.json()

        # transform list of lists to pandas dataframe and return
        return pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from Coinbase API: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def get_last_ingested_timestamp(feature_group) -> datetime:
    """
    Retrieve the exact last ingested timestamp from the feature group.
    """
    data = feature_group.read()
    if data.empty:
        return None

    # Get the last ingested timestamp
    last_timestamp = data['time'].iloc[-1]

    # Convert to datetime object
    return pd.to_datetime(last_timestamp, unit='s')


def download_data_since_last_ingested(feature_group) -> pd.DataFrame:
    """
    Download data from Coinbase since the last ingested timestamp.
    """
    last_ingested_timestamp = get_last_ingested_timestamp(feature_group)
    if last_ingested_timestamp is None:
        raise ValueError("No previous data ingested.")

    # Convert last ingested timestamp to the next hour
    from_time = last_ingested_timestamp + timedelta(hours=1)
    from_time_str = from_time.strftime('%Y-%m-%dT%H:%M:%S')
    print('starting time frame frame of ingested data is:',from_time_str)
    to_time = datetime.utcnow()
    to_time_str = to_time.strftime('%Y-%m-%dT%H:%M:%S')

    URL = f'https://api.exchange.coinbase.com/products/DOGE-USD/candles?start={from_time_str}&end={to_time_str}&granularity=3600'
    r = requests.get(URL)
    data = r.json()

    return pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

if __name__ == '__main__':
    try:
        connection = hopsworks.login(project=project_name, api_key_value=api_key)
        fs = connection.get_feature_store()
        dogecoin_fg = fs.get_or_create_feature_group(name="dogecoin",
                                                 version=1,
                                                 primary_key=["time"],
                                                 description="OHLC data of Dogecoin")
        dogecoin_df = download_data_since_last_ingested(dogecoin_fg)
        # print(dogecoin_df)
        if dogecoin_df is None or dogecoin_df.empty:
            print("No new data ingested. Existing data is up-to-date.")
        else:
            dogecoin_df['volume'] = dogecoin_df['volume'].astype('double')
            dogecoin_fg.insert(dogecoin_df)
    except hopsworks.client.exceptions.RestAPIError as e:
        logger.error(f"Error connecting to Hopsworks API: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

    # if BACKFILL==False:
    #     data=download_last_hour_data("DOGE-USD")
    #     if data.empty:
    #         logger.info('No new data fetched for the last hour, exiting.')
    #         exit()
    #     else:
    #         logger.info('New data fetched, proceeding with insertion.')
    # else:
    #     dogecoin=download_ohlc_data_from_coinbase("DOGE-USD","2021-06-04","2024-07-14")
    #
    # try:
    #     connection = hopsworks.login(project=project_name, api_key_value=api_key)
    #     fs = connection.get_feature_store()
    #
    #     if BACKFILL == False:
    #         dogecoin_df = data
    #     else:
    #         dogecoin_df = pd.read_parquet(dogecoin)
    #
    #     dogecoin_df['volume'] = dogecoin_df['volume'].astype('double')
    #     dogecoin_fg = fs.get_or_create_feature_group(name="dogecoin",
    #                                                  version=1,
    #                                                  primary_key=["time"],
    #                                                  description="OHLC data of Dogecoin")
    #     dogecoin_fg.insert(dogecoin_df)
    # except hopsworks.client.exceptions.RestAPIError as e:
    #     logger.error(f"Error connecting to Hopsworks API: {str(e)}")
    #     exit(1)
    # except Exception as e:
    #     logger.error(f"Unexpected error: {str(e)}")
    #     exit(1)