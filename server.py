import time
import litserve as ls
from src.api.llama_vision import LlamaVisionAPI


class Logger(ls.Logger):
    def __init__(self):
        super().__init__()

    def process(self, key, value):
        if key == "input_text":
            print(f"Input text: {value}")

        if key == "output_text":
            print(value, end="", flush=True)

        if key == "inference_time":
            print(f"Inference time: {value:.3f} seconds")


class InferenceTimeLogger(ls.Callback):
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
        callbacks=[InferenceTimeLogger()],
        loggers=Logger(),
        timeout=60,
    )
    server.run(port=8000, generate_client_file=False)
