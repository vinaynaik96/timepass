import requests

def get_node_exporter_metrics(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching metrics: {e}")
        return None

def calculate_cpu_usage(metrics_text):
    try:
        cpu_metrics = [line for line in metrics_text.splitlines() if line.startswith('cpu ')]
        if not cpu_metrics:
            raise ValueError("Failed to find 'cpu' metrics in the response.")
        
        # Parse the CPU metrics
        cpu_metrics = cpu_metrics[0].split()[1:]

        # Calculate total CPU time
        total_cpu_time = sum(map(int, cpu_metrics))

        # Calculate idle CPU time
        idle_cpu_time = int(cpu_metrics[3])

        # Calculate CPU usage percentage
        cpu_usage_percentage = 100.0 * (1.0 - idle_cpu_time / total_cpu_time)
        return cpu_usage_percentage
    except (IndexError, ValueError) as e:
        print(f"Error occurred while calculating CPU usage: {e}")
        return None

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
