import argparse
from src.api import client
from src.ui.utils import encode_image
from src.config import MODEL


def send_generate_request(image_path, prompt):
    encoded_image_object = encode_image(image_path)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    encoded_image_object,
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        print(
            f"\033[92m{chunk.choices[0].delta.content or ''}\033[0m", end="", flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send text & image to Llama 3.2 Vision API server and receive a response."
    )
    parser.add_argument("--image", required=True, help="Path for the image file")
    parser.add_argument("--prompt", required=True, help="Prompt about the image")
    args = parser.parse_args()

    send_generate_request(args.image, args.prompt)
