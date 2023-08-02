def send_alert(data):
    url_alert = 'https://cognizantcri.service-now.com/api/now/table/em_alert'
    url_incident = 'https://cognizantcri.service-now.com/api/now/table/incident'
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

    incident_details = {
        "short_description": "High CPU Utilization on Server123",
        "description": f"CPU Utilization is high on Server123, current value is {data}",
        "priority": "1",
        "severity": "1",
        "state": "New",
        "category": "Hardware",
        "subcategory": "CPU"
    }

    headers = {"Content-Type":"application/json","Accept":"application/json"}

    # Create alert
    response_alert = requests.post(url_alert, auth=(user, pwd), headers=headers, data=json.dumps(alert_details))
    if response_alert.status_code == 200: 
        print('Status:', response_alert.status_code, 'Headers:', response_alert.headers, 'Error Response:',response_alert.json())
    response_data_alert = response_alert.json()

    if response_alert.status_code == 200 or response_alert.status_code == 201:
        alert_number = response_data_alert['result']['sys_id']  # replace 'sys_id' with the actual field name
        print(f"Alert is created for CPU value {data} at {time.ctime()}, and alert number is {alert_number}")
    else:
        print(response_data_alert)

    # Create incident
    response_incident = requests.post(url_incident, auth=(user, pwd), headers=headers, data=json.dumps(incident_details))
    response_data_incident = response_incident.json()

    if response_incident.status_code == 200 or response_incident.status_code == 201:
        incident_number = response_data_incident['result']['number']  # replace 'number' with the actual field name
        print(f"Incident is created for CPU value {data} at {time.ctime()}, and incident number is {incident_number}")
    else:
        print(response_data_incident)
