import litserve as ls
from src.api.llama_vision import LlamaVisionAPI


if __name__ == "__main__":
    api = LlamaVisionAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000, generate_client_file=False)
