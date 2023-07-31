import time
from prometheus_api_client import PrometheusConnect

def get_cpu_usage(prometheus_url):
    prom = PrometheusConnect(url=prometheus_url)
    query = '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
    try:
        result = prom.custom_query(query)
        if result:
            cpu_usage = result[0]['value'][1]
            return float(cpu_usage)
        else:
            print("Error: No data returned from Prometheus.")
    except Exception as e:
        print(f"Error: {e}")

    return None

if __name__ == "__main__":
    prometheus_url = "http://your_prometheus_url/"
    while True:
        cpu_usage = get_cpu_usage(prometheus_url)
        if cpu_usage is not None:
            print(f"CPU Usage: {cpu_usage:.2f}%")
        else:
            print("Unable to retrieve CPU usage.")
        time.sleep(1)  # Adjust the interval as needed (in seconds).
