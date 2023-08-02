import requests
import time

def moniter_prometheus():
    prome_sql = """(sum by(instance) (irate(node_cpu_seconds_total{mode!="idle"}[1m])) / on(instance) group_left sum by (instance)((irate(node_cpu_seconds_total[1m])))) * 100"""
    url= 'http://54.162.54.22:9090/api/v1/query'
    params = {'query': prome_sql}
    
    while True:
        try:
            response = requests.get(url,params=params)
            data = response.json()["data"]['result'][0]['value'][1]
            print(f"current value:{data}")
                  
        except:
            print(f"error: {e}")
        time.sleep(1)    
                  
if __name__=="__main__":
    moniter_prometheus()




import requests
import json
url = 'https://cognizantcri.service-now.com/api/now/table/em_alert'
user = "229785"
pwd = "Welcome@123"

alert_details = {
    "alert_id": "ALERT12345",
    "source": "Monitoring System",
    "node": "Server123",
    "type": "CPU Utilization High",
    "resource": "CPU",
    "metric_name": "CPU Utilization",
    "severity": "1",
    "description": "CPU Utilization is high on Server123",
    "state": "open",
    "additional_info": "CPU Utilization has been above 90% for the last 15 minutes",
    "short_description": "High CPU Utilization on Server123"
}

headers = {"Content-Type":"application/json","Accept":"application/json"}
response = requests.post(url, auth=(user, pwd), headers=headers, data=json.dumps(alert_details) )
if response.status_code == 200: 
    print('Status:', response.status_code, 'Headers:', response.headers, 'Error Response:',response.json())
data = response.json()
print(data)
