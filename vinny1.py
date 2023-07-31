import requests

def get_node_exporter_metrics(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch metrics. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching metrics: {e}")
        return None

def calculate_cpu_usage(metrics_text):
    cpu_metrics = [line for line in metrics_text.splitlines() if line.startswith('cpu ')]
    if not cpu_metrics:
        return None

    # Parse the CPU metrics
    cpu_metrics = cpu_metrics[0].split()[1:]

    # Calculate total CPU time
    total_cpu_time = sum(map(int, cpu_metrics))

    # Calculate idle CPU time
    idle_cpu_time = int(cpu_metrics[3])

    # Calculate CPU usage percentage
    cpu_usage_percentage = 100.0 * (1.0 - idle_cpu_time / total_cpu_time)
    return cpu_usage_percentage

if __name__ == "__main__":
    prometheus_node_exporter_url = "http://localhost:9001/metrics"

    metrics_text = get_node_exporter_metrics(prometheus_node_exporter_url)
    if metrics_text:
        cpu_usage = calculate_cpu_usage(metrics_text)
        if cpu_usage is not None:
            print(f"CPU Usage: {cpu_usage:.2f}%")
        else:
            print("Failed to calculate CPU usage.")
    else:
        print("Failed to fetch metrics from Prometheus Node Exporter.")
