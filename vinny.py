import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import requests
import time
import datetime, pytz

def fetch_data_from_db():
    password = "merck@1234"
    encoded_password = urllib.parse.quote(password, safe='')
    engine = create_engine(f"postgresql://merck_svc:{encoded_password}@54.205.22.2/merck_db")
    
    query = "SELECT * FROM prediction_result1"
    df = pd.read_sql_query(query, engine)

    return df

def send_alert(data):
    utc_time = datetime.datetime.now(pytz.timezone('UTC')).strftime("%I:%M:%S %p")
    return f"The CPU Utilization is {data} and Alert_Number:123456 at {utc_time}"

def difference(actual, predicted):
    return abs(actual - predicted)

def monitor_prometheus(df, threshold=30):
    prome_sql = """(sum by(instance) (irate(node_cpu_seconds_total{mode!="idle"}[1m])) / on(instance) 
                    group_left sum by (instance)((irate(node_cpu_seconds_total[1m])))) * 100"""
    url= 'http://54.162.54.22:9090/api/v1/query'
    params = {'query': prome_sql}
    
    while True:
        try:
            response = requests.get(url,params=params)
            data = float(response.json()["data"]['result'][0]['value'][1])
            print(f"current value: {data}")
            if data > threshold:
                alert_message = send_alert(data)
                print(alert_message)
                # Fetch data from database again
                df = fetch_data_from_db()
                # Check if the predicted CPU value is close to the actual value
                predicted_cpu = df['predicted_cpu'].iloc[-1]  # get the latest predicted value
                diff = difference(data, predicted_cpu)
                if diff <= 10:  # compare with a tolerance of 10 units
                    print(f"Incident created. Difference between actual CPU value {data} and predicted CPU value {predicted_cpu} is {diff}, which is within the allowed limit.")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(60)

if __name__=="__main__":
    df = fetch_data_from_db()
    now = datetime.datetime.now()
    delay = 60 - now.second
    time.sleep(delay)
    monitor_prometheus(df)
