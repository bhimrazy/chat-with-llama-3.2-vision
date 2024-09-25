import argparse
import requests

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:8000/predict"


def send_generate_request(image_path, prompt):
    with open(image_path, "rb") as f:
        image_file = f.read()

    files = {"image": (image_path, image_file, "image/jpeg")}
    response = requests.post(API_URL, data={"prompt": prompt}, files=files, stream=True)
    if response.status_code == 200:
        for line in response.iter_lines():
            print(f"\033[92m{line.decode('utf-8')}\033[0m")
    else:
        print(
            f"Error: Response with status code {response.status_code} - {response.text}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send text & image to Llama 3.2 Vision API server and receive a response."
    )
    parser.add_argument("--image", required=True, help="Path for the image file")
    parser.add_argument("--prompt", required=True, help="Prompt about the image")
    args = parser.parse_args()

    send_generate_request(args.image, args.prompt)
