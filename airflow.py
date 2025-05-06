import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator 
import os

from model import train 

Data_url = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption.zip"
Data_file = "power_consumption.zip"
Unz_data_file = "household_power_consumption.txt"
Clean_data = "power_consumption_cleaned.csv" 

def download_data():
    os.system(f"curl -o {Data_file} {Data_url}") 
    print(f"Data downloaded successfully to {Data_file}")
    return True

def unzip_data():
    os.system(f"unzip {Data_file}")
    print(f"Data unzipped successfully")
    return True

def clean_data():
    df = pd.read_csv(Unz_data_file, sep=';', na_values=['?']) 
    df.dropna(inplace=True)
       
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df['Global_reactive_power'] = df['Global_reactive_power'].astype(float)
    df['Voltage'] = df['Voltage'].astype(float)
    df['Global_intensity'] = df['Global_intensity'].astype(float)
    df['Sub_metering_1'] = df['Sub_metering_1'].astype(float)
    df['Sub_metering_2'] = df['Sub_metering_2'].astype(float)
    df['Sub_metering_3'] = df['Sub_metering_3'].astype(float)

    df.to_csv(Clean_data, index=False)
    print(f"Data cleaned and saved to {Clean_data}")
    return True

dag_power = DAG(
    dag_id="power_consumption_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  
    catchup=False,
    default_args={
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    tags=['power_consumption', 'ml']
)

download_task = PythonOperator(task_id="download_power_data", python_callable=download_data, dag=dag_power, )
unzip_task = PythonOperator(task_id="unzip_power_data", python_callable=unzip_data, dag=dag_power)
clean_task = PythonOperator(task_id="clean_power_data", python_callable=clean_data, dag=dag_power, )
train_task = PythonOperator(task_id="train_power_model", python_callable=train,  dag=dag_power, )

download_task >> unzip_task >> clean_task >> train_task 
