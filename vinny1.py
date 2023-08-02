url = 'https://cognizantcri.service-now.com/api/now/table/incident?sysparm_limit=1'
user = "229785"
pwd = "Welcome@123"
short_description = "MONITORING TEST: Share is full"
assignment_group = "0996f38ec89112009d04d87a50caf610"
contact_type = "Event1"
u_contact = "a7761d3a44511200e4852779951f55da"  # you / contact person
caller_id = "a7761d3a44511200e4852779951f55da"  # affected user
u_creator_group = "0996f38ec89112009d04d87a50caf610"
description = "Share 'this and this' is full, please do something."
u_symptom = "b3a47ffcb07932002f10272c5c585dfc"  # Information
state = '3'  # Assigned, 3 - work in progress, 2 - assigned
incident_state = '2'  # Assigned ?
u_infrastructure_ci = '42e2472a65fa26009d04fcdf1618cb81'  # Dummy CI
work_notes = 'a work notes test1'  # Work notes
comments = 'comments test'  # Additional comments

 

# Set proper headers
headers = {"Content-Type": "application/json", "Accept": "application/json"}

 

# Do the HTTP request
response = requests.post(url, auth=(user, pwd), headers=headers,
                         data=str({"short_description": short_description,
                                   "u_creator_group": u_creator_group,
                                   "contact_type": contact_type,
                                   "u_contact": u_contact,
                                   "description": description,
                                   "u_infrastructure_ci": u_infrastructure_ci,
                                   "u_symptom": u_symptom,
                                   "caller_id": caller_id,
                                   "work_notes": work_notes,
                                   "comments": comments,
                                   "assignment_group": assignment_group,
                                   "state": state,
                                   }))

 

response.text
