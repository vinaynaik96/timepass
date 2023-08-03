import requests
import time
import json
import datetime

def monitor_prometheus(threshold=30):
    prome_sql = """(sum by(instance) (irate(node_cpu_seconds_total{mode!="idle"}[1m])) / on(instance) group_left sum by (instance)((irate(node_cpu_seconds_total[1m])))) * 100"""
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
    return f"The CPU Utilization is {data} and Alert Number : 123456"
    
if __name__=="__main__":
    # delay the start of the script until the next minute begins
    now = datetime.datetime.now()
    delay = 60 - now.second
    time.sleep(delay)
    monitor_prometheus()
