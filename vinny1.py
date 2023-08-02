import requests
import time
import json

def monitor_prometheus(threshold=9):
    prome_sql = """(sum by(instance) (irate(node_cpu_seconds_total{mode!="idle"}[1m])) / on(instance) group_left sum by (instance)((irate(node_cpu_seconds_total[1m])))) * 100"""
    url= 'http://54.162.54.22:9090/api/v1/query'
    params = {'query': prome_sql}
    
    while True:
        try:
            response = requests.get(url,params=params)
            data = float(response.json()["data"]['result'][0]['value'][1])
            print(f"current value: {data}")
            
            if data > threshold:
                send_alert(data)
                
        except Exception as e:
            print(f"error: {e}")
        time.sleep(1)

def send_alert(data):
    url = 'https://cognizantcri.service-now.com/api/now/table/em_alert'
    user = "username"
    pwd = "password"

    alert_details = {
        "alert_id": "ALERT12345",
        "source": "Monitoring System",
        "node": "Server123",
        "type": "CPU Utilization High",
        "resource": "CPU",
        "metric_name": "CPU Utilization",
        "severity": "1",
        "description": f"CPU Utilization is high on Server123, current value is {data}",
        "state": "open",
        "additional_info": f"CPU Utilization has been above 90% for the last 15 minutes, current value is {data}",
        "short_description": "High CPU Utilization on Server123"
    }

    headers = {"Content-Type":"application/json","Accept":"application/json"}
    response = requests.post(url, auth=(user, pwd), headers=headers, data=json.dumps(alert_details))
    if response.status_code == 200: 
        print('Status:', response.status_code, 'Headers:', response.headers, 'Error Response:',response.json())
    data = response.json()
    print(data)

if __name__=="__main__":
    monitor_prometheus()
def send_alert(data):
    url = 'https://cognizantcri.service-now.com/api/now/table/em_alert'
    user = "username"
    pwd = "password"

    alert_details = {
        "alert_id": "ALERT12345",
        "source": "Monitoring System",
        "node": "Server123",
        "type": "CPU Utilization High",
        "resource": "CPU",
        "metric_name": "CPU Utilization",
        "severity": "1",
        "description": f"CPU Utilization is high on Server123, current value is {data}",
        "state": "open",
        "additional_info": f"CPU Utilization has been above 90% for the last 15 minutes, current value is {data}",
        "short_description": "High CPU Utilization on Server123"
    }

    headers = {"Content-Type":"application/json","Accept":"application/json"}
    response = requests.post(url, auth=(user, pwd), headers=headers, data=json.dumps(alert_details))
    if response.status_code == 200: 
        print('Status:', response.status_code, 'Headers:', response.headers, 'Error Response:',response.json())
    response_data = response.json()

    if response.status_code == 200 or response.status_code == 201:
        alert_number = response_data['result']['sys_id']  # replace 'sys_id' with the actual field name
        print(f"Alert is created for CPU value {data} at {time.ctime()}, and alert number is {alert_number}")
    else:
        print(response_data)
