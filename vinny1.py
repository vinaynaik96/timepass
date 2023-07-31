from prometheus_client import start_http_server, Summary, Gauge
import requests
import time

# Define the Prometheus Node Exporter endpoint URL
PROMETHEUS_ENDPOINT = "http://localhost:9001/metrics"

# Define a custom function to fetch the metrics from the Prometheus endpoint
def get_metrics(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch metrics. Status code: {response.status_code}")

# Parse the Prometheus metrics to get the CPU usage
def get_cpu_usage(metrics):
    cpu_usage = None
    for line in metrics.splitlines():
        if line.startswith("node_cpu_seconds_total"):
            _, _, mode = line.split("{")[1].strip("}").split(",")
            if mode == 'mode="idle"':
                cpu_usage = 100.0 - float(line.split()[-1])
    return cpu_usage

if __name__ == '__main__':
    # Start an HTTP server to expose the metrics (optional)
    start_http_server(8000)

    # Infinite loop to continuously monitor CPU usage
    while True:
        try:
            metrics = get_metrics(PROMETHEUS_ENDPOINT)
            cpu_usage = get_cpu_usage(metrics)

            if cpu_usage is not None:
                print(f"CPU Usage: {cpu_usage:.2f}%")
                # You can perform additional actions with the CPU usage value here

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)  # Adjust the interval as needed
