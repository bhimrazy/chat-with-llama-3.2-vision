import base64
import concurrent.futures
import json
import re
from io import BytesIO

import requests
from litserve.specs.openai import ChatCompletionRequest, ResponseFormat, Tool
from PIL import Image
from typing import List


def read_image(source):
    """
    Read an image from a real image URL or a base64-encoded URL.

    Parameters:
    source (str): The image source. Can be a real image URL or a base64 URL string.

    Returns:
    Image or None: The Image object if the source is valid, otherwise None.
    """
    try:
        if re.match(r"^https?://", source):
            # It's a real image URL
            return Image.open(requests.get(source, stream=True).raw).convert("RGB")
        elif re.match(r"^data:image/.+;base64,", source):
            # It's a base64 image URL
            base64_image = source.split(",")[1]
            image_data = base64.b64decode(base64_image)
            return Image.open(BytesIO(image_data)).convert("RGB")
        else:
            return Image.open(source).convert("RGB")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def prep_tool_prompt(tools: List[Tool]):
    """
    Prepare system prompt with tools.
    """

    function_definitions = [tool.model_dump(exclude_none=True) for tool in tools]
    system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.\n\n{functions}\n""".format(
        functions=json.dumps(function_definitions, indent=4)
    )

    return system_prompt


def prep_schema_prompt(system_prompt: str, response_format: ResponseFormat):
    """
    Prepare system prompt with response format.

    response format prompt adapted from : https://github.com/SylphAI-Inc/AdalFlow
    """
    response_format_str = ""
    response_format = response_format.model_dump(exclude_none=True, by_alias=True)
    schema = response_format.get("json_schema", None)

    if schema:
        response_format_str = (
            "<RESPONSE_FORMAT>\n"
            "Your output should be formatted as a standard JSON instance with the following schema:\n"
            "```\n"
            f"{json.dumps(schema, indent=4)}\n"
            "```\n"
            "- Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!\n"
            "- Use double quotes for the keys and string values.\n"
            '- DO NOT mistake the "properties" and "type" in the schema as the actual fields in the JSON output.\n'
            "- DO NOT include any additional text, comments, or annotations outside of the JSON object.\n"
            "- Follow the structure and field names as specified in the schema exactly.\n"
            "- Follow the JSON formatting conventions.\n"
            "- DO NOT include schema definitions in the JSON output.\n"
            "- Ensure that the JSON output strictly conforms to the schema provided without deviation.\n"
            "- Do validate your JSON output for syntax correctness and adherence to the schema before submission.\n"
            "- Strictly adhere to the schema provided above.\n"
            "- Return the markdown JSON object as the output without any additional text or comments.\n"
            "</RESPONSE_FORMAT>"
        )
    else:
        response_format_str = (
            "<RESPONSE_FORMAT>\n"
            "Your output should be formatted as a standard JSON instance.\n"
            "- Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!\n"
            "- Use double quotes for the keys and string values.\n"
            '- DO NOT mistake the "properties" and "type" in the schema as the actual fields in the JSON output.\n'
            "- Follow the JSON formatting conventions.\n"
            "</RESPONSE_FORMAT>"
        )
    return f"{system_prompt}\n\n{response_format_str}"


def parse_messages(request: ChatCompletionRequest):
    """
    Parse messages from a ChatCompletionRequest object.
    """
    messages = []
    images = []
    response_format = request.response_format
    tools = request.tools

    for message in request.messages:
        content = message.content
        if message.role == "system":
            if tools:
                content = prep_tool_prompt(tools)
            if response_format:
                content = prep_schema_prompt(content, response_format)

        if isinstance(content, list):
            content = []
            for content_item in message.content:
                if content_item.type == "image_url":
                    url = content_item.image_url.url
                    images.append(url)
                    content.append({"type": "image"})
                elif content_item.type == "text":
                    content.append({"type": "text", "text": content_item.text})

        messages.append({"role": message.role, "content": content})

    def process_image(image_url):
        image = read_image(image_url)
        # resize if height is greater than 720
        if image.height > 720:
            image = image.resize((int(image.width * 720 / image.height), 720))
        return image

    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, images))

    return messages, images
