import litserve as ls


class SimpleStreamAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: x

    def decode_request(self, request):
        print(request)
        # prompt = request.get("prompt")
        # # image = request.get("image").file
        # print(prompt)
        return "input"

    def predict(self, x):
        for i in range(10):
            yield self.model(i)

    def encode_response(self, output):
        for out in output:
            yield {"output": out}


if __name__ == "__main__":
    api = SimpleStreamAPI()
    server = ls.LitServer(api, stream=True)
    server.run(port=8000)
