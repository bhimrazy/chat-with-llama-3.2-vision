from threading import Thread

import litserve as ls
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    TextIteratorStreamer,
)
from PIL import Image


class LlamaVisionAPI(ls.LitAPI):
    def setup(self, device):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quantization_config,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.device = device
        self.model_id = model_id

    def decode_request(self, request):
        prompt = request.get("prompt")
        image = request.get("image").file
        image = Image.open(image).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.device)
        return inputs

    def predict(self, inputs):
        generation_kwargs = dict(
            **inputs,
            streamer=self.streamer,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=1024,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in self.streamer:
            yield text

    def encode_response(self, outputs):
        for output in outputs:
            yield output


if __name__ == "__main__":
    api = LlamaVisionAPI()
    server = ls.LitServer(api, api_path="/predict", stream=True)
    server.run(port=8000, generate_client_file=False)
