# Chat with Llama 3.2-Vision (11B) Multimodal LLM
<div align="center">
<img src="https://github.com/user-attachments/assets/645d4447-eb8a-4992-9c53-8c37e904e82f" style="display: block; margin-left: auto; margin-right: auto; border-radius: 50%; width: 128px; height: 128px; object-fit: cover; "><br/><br/>
<a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-and-chat-with-llama-3-2-vision-multimodal-llm-using-litserve-lightning-fast-inference-engine">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a> 
</div>


## Overview

The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. [Read more](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

## Features

- Visual recognition
- Image reasoning
- Captioning
- Answering general questions about an image
- Tool Calling

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/bhimrazy/chat-with-llama-3.2-vision
    cd chat-with-llama-3.2-vision
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run server
    ```sh
    export HF_TOKEN=your_huggingface_token # required for model download

    python server.py
    ```

2. Run client/app

    To test using python client, execute the following command:

<div align="center">
    <img src="cocktail-ingredients.jpg" height="200" width="auto"><br>
    <i>What cocktail can I make with these ingredients?</i>
</div>


```sh
python client.py --image=cocktail-ingredients.jpg --prompt="What cocktail can I make with these ingredients?"
```
To run the application, execute the following command:
```sh
streamlit run app.py
```

