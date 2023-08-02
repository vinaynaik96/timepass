import requests
import json

# Variables
url = 'https://cognizantcri.service-now.com/api/now/table/incident?sysparm_limit=1'
username = "229785"
password = "Welcome@123"

# Incident details
incident_details = {
    "short_description": "MONITORING TEST: Share is full",
    "assignment_group": "0996f38ec89112009d04d87a50caf610",
    "contact_type": "Event1",
    "u_contact": "a7761d3a44511200e4852779951f55da",  # contact person
    "caller_id": "a7761d3a44511200e4852779951f55da",  # affected user
    "u_creator_group": "0996f38ec89112009d04d87a50caf610",
    "description": "Share 'this and this' is full, please do something.",
    "u_symptom": "b3a47ffcb07932002f10272c5c585dfc",  # Information
    "state": '3',  # Assigned, 3 - work in progress, 2 - assigned
    "incident_state": '2',  # Assigned ?
    "u_infrastructure_ci": '42e2472a65fa26009d04fcdf1618cb81',  # Dummy CI
    "work_notes": 'a work notes test1',  # Work notes
    "comments": 'comments test'  # Additional comments
}

# Set headers
headers = {"Content-Type": "application/json", "Accept": "application/json"}

# Do the HTTP request
response = requests.post(
    url,
    auth=(username, password),
    headers=headers,
    data=json.dumps(incident_details)  # Convert dictionary to json
)

# Check the response
if response.status_code != 200:
    print(f"Error: Got status code {response.status_code} from ServiceNow API. Incident may not have been created successfully.")
else:
    print("Incident created successfully.")

# Print the response
print(response.text)
