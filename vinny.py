import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import requests
import time
import datetime, pytz
from scipy.stats import pearsonr

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
                # Check correlation
                correlation, _ = pearsonr(df['actual_cpu'], df['predicted_cpu'])  # replace with actual column names
                if correlation > 0.5:  # replace with desired correlation threshold
                    trace_result = df['trace_result']  # replace with actual column name
                    print(f"Incident created. Trace result: {trace_result}")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(60)

if __name__=="__main__":
    df = fetch_data_from_db()
    now = datetime.datetime.now()
    delay = 60 - now.second
    time.sleep(delay)
    monitor_prometheus(df)
