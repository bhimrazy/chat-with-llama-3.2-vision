import os
import time
import litserve as ls
from src.api.llama_vision import LlamaVisionAPI
from prometheus_client import Summary


def generate_metrics_dir():
    # Generate a metrics directory for Prometheus metrics if not already set
    metrics_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")

    if not metrics_dir:
        # Create a temp directory if it is not set
        metrics_dir = os.path.join(os.getcwd(), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
        print(f"PROMETHEUS_MULTIPROC_DIR set to: {metrics_dir}")
    else:
        # If the directory exists, ensure it is valid
        if not os.path.isdir(metrics_dir):
            raise ValueError(f"{metrics_dir} is not a valid directory")
        print(f"PROMETHEUS_MULTIPROC_DIR already set to: {metrics_dir}")

    return metrics_dir


class PrometheusLogger(ls.Logger):
    def __init__(self):
        super().__init__()
        # self.summary = Summary(
        #     "inference_speed",
        #     documentation="Inference speed in seconds",
        # )

    def process(self, key, value):
        if key == "input_text":
            print(f"Input text: {value}")

        if key == "output_text":
            print(value, end="", flush=True)

        if key == "inference_time":
            print(f"Inference time: {value:.3f} seconds")


class InferenceSpeedLogger(ls.Callback):
    def on_before_predict(self, lit_api):
        t0 = time.perf_counter()
        self._start_time = t0

    def on_after_predict(self, lit_api):
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time
        lit_api.log("inference_time", elapsed)


if __name__ == "__main__":
    api = LlamaVisionAPI()

    server = ls.LitServer(
        api,
        spec=ls.OpenAISpec(),
        timeout=60,
        callbacks=[InferenceSpeedLogger()],
        loggers=PrometheusLogger(),
    )
    server.run(port=8000, generate_client_file=False)
