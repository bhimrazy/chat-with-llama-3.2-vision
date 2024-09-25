import argparse
import requests

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:8000/predict"


def send_generate_request(image_path, prompt):
    with open(image_path, "rb") as f:
        image = f.read()
    response = requests.post(API_URL, files={"image": image}, data={"prompt": prompt})
    if response.status_code == 200:
        print(f"Response: {response.text}")
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
