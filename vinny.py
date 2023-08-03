
import pandas as pd
from sqlalchemy import create_engine
import urllib

def fetch_data_from_db():
    password = "merck@1234"
    encoded_password = urllib.parse.quote(password, safe='')
    engine = create_engine(f"postgresql://merck_svc:{encoded_password}@localhost/merck_db")
    
    query = "SELECT * FROM prediction_result1"
    df = pd.read_sql_query(query, engine)

    return df


import requests
import time
import json
import datetime, pytz

def monitor_prometheus(threshold=30):
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
                print(send_alert(data))        
        except Exception as e:
            print(f"error: {e}")
        time.sleep(60)

def send_alert(data):
    utc_time = datetime.datetime.now(pytz.timezone('UTC')).strftime("%I:%M:%S %p")
    return f"The CPU Utilization is {data} and Alert_Number:123456 at {utc_time}"
    
if __name__=="__main__":
    now = datetime.datetime.now()
    delay = 60 - now.second
    time.sleep(delay)
    monitor_prometheus()
