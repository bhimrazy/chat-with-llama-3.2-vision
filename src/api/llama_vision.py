from threading import Thread

import litserve as ls
import torch
from litserve.specs.openai import ChatCompletionRequest, ChatMessage
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    TextIteratorStreamer,
)

from src.api.utils import parse_messages
from src.config import MODEL
from src.tools.tool_utils import ToolUtils


class LlamaVisionAPI(ls.LitAPI):
    def setup(self, device):
        model_id = MODEL
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            # quantization_config=quantization_config,
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

    def decode_request(self, request: ChatCompletionRequest, context: dict):
        context["generation_args"] = {
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "max_new_tokens": request.max_tokens or 2048,
        }
        context["tool"] = request.tools is not None

        messages, images = parse_messages(request)
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        self.log("input_text", input_text)
        inputs = self.processor(images, input_text, return_tensors="pt").to(self.device)
        return inputs

    def predict(self, inputs, context: dict):
        generation_kwargs = dict(
            **inputs,
            streamer=self.streamer,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **context["generation_args"],
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in self.streamer:
            yield text

    def encode_response(self, outputs, context: dict):
        buffer = []
        for output in outputs:
            buffer.append(output)
            self.log("output_text", output)

            # Check if the output could be a tool call
            combined_output = "".join(buffer).strip()
            if context.get("tool") and combined_output.startswith(
                ("{", "[", "<function")
            ):
                tool_calls = ToolUtils.maybe_extract_custom_tool_calls(combined_output)
                yield ChatMessage(role="assistant", content="", tool_calls=tool_calls)
                continue

            # Handle end-of-sequence (EOS) token
            if self.processor.tokenizer.eos_token in output:
                output = output.replace(self.processor.tokenizer.eos_token, "")

            yield ChatMessage(role="assistant", content=output)
