import time
import requests

def get_cpu_usage(node_exporter_url):
    try:
        response = requests.get(node_exporter_url)
        if response.status_code == 200:
            metrics = response.text.splitlines()
            for metric in metrics:
                if metric.startswith('node_cpu_seconds_total{cpu="0", mode="idle"}'):
                    _, cpu_usage = metric.split(' ')
                    return float(cpu_usage)
        else:
            print(f"Error: Unable to fetch data from {node_exporter_url}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return None

if __name__ == "__main__":
    node_exporter_url = "http://your_node_exporter_endpoint/metrics"
    while True:
        cpu_usage = get_cpu_usage(node_exporter_url)
        if cpu_usage is not None:
            print(f"CPU Usage: {cpu_usage:.2f}%")
        else:
            print("Unable to retrieve CPU usage.")
        time.sleep(1)  # Adjust the interval as needed (in seconds).
