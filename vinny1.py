import requests

prome_sql = """(sum by(instance) (irate(node_cpu_seconds_total{mode!="idle"}[1m])) / on(instance) group_left sum by (instance)((irate(node_cpu_seconds_total[1m])))) * 100"""

response = requests.get('http://54.162.54.22:9090/api/v1/query',
    params={'query': prome_sql})
print(response.json()["data"]['result'][0]['value'][1])
