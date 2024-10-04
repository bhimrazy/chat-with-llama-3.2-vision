import base64
import concurrent.futures
import json
import re
from io import BytesIO

import requests
from litserve.specs.openai import (
    ChatCompletionRequest,
    ResponseFormat,
    Tool,
    ChatMessage,
)
from PIL import Image
from typing import List, Union, Dict


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

    function_definitions = [
        tool.function.model_dump(exclude_none=True) for tool in tools
    ]
    system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
MAKE SURE the function call is in the correct format along with the correct/valid parameters.
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.\n\n{functions}\n""".format(
        functions=json.dumps(function_definitions, indent=4)
    )

    return system_prompt


def prep_schema_prompt(
    system_prompt: str, response_format: Union[ResponseFormat, Dict]
):
    """
    Prepare system prompt with response format.

    response format prompt adapted from : https://github.com/SylphAI-Inc/AdalFlow
    """
    response_format_str = ""
    response_format = (
        response_format.model_dump(exclude_none=True, by_alias=True)
        if isinstance(response_format, ResponseFormat)
        else response_format
    )
    schema = response_format.get("json_schema", None)

    if schema:
        response_format_str = (
            "<RESPONSE_FORMAT>\n"
            "Your output should be formatted as a standard JSON instance with the following schema:\n"
            "```\n"
            f"{json.dumps(schema, indent=4)}\n"
            "```\n"
            "- Always enclose the JSON output in triple backticks (```).\n"
            "- Ensure that only valid JSON output is included, without any additional text or formatting.\n"
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
            "- Always enclose the JSON output in triple backticks (```).\n"
            "- Ensure that only valid JSON output is included, without any additional text or formatting.\n"
            "- Use double quotes for the keys and string values.\n"
            '- DO NOT mistake the "properties" and "type" in the schema as the actual fields in the JSON output.\n'
            "- Follow the JSON formatting conventions.\n"
            "</RESPONSE_FORMAT>"
        )
    return f"{system_prompt}\n\n{response_format_str}"


def process_image(image_url: str) -> Image:  # type: ignore
    """
    Process an image: read and resize if its height is greater than 720.
    """
    image = read_image(image_url)
    if image and image.height > 720:
        image = image.resize((int(image.width * 720 / image.height), 720))
    return image  # type: ignore


def process_content(
    content: Union[str, List],
    message: ChatMessage,
    tools: List[Tool] | None,
    images: List,
    response_format: ResponseFormat | None,
    last_user_message: bool,
) -> Union[str, List[Dict]]:
    """
    Process the content of a message based on its type and other conditions.
    """
    if message.role == "system" and tools:
        content = prep_tool_prompt(tools)

    if last_user_message and response_format and isinstance(content, str):
        content = prep_schema_prompt(content, response_format)

    if isinstance(content, list):
        content = process_content_list(content, images)

        if last_user_message and response_format:
            content.append(
                {"type": "text", "text": prep_schema_prompt("", response_format)}
            )

    return content


def process_content_list(
    content_list: List,
    images: List,
) -> List[Dict]:
    """
    Process a list of content items.
    """
    content = []
    for content_item in content_list:
        if content_item.type == "image_url":
            url = content_item.image_url.url
            images.append(url)
            content.append({"type": "image"})
        elif content_item.type == "text":
            content.append({"type": "text", "text": content_item.text})
    return content


def parse_messages(request: ChatCompletionRequest):
    """
    Parse messages from a ChatCompletionRequest object.
    """
    messages = []
    images = []
    response_format = request.response_format
    tools = request.tools

    total_messages = len(request.messages)
    total_messages = len(request.messages)
    for i, message in enumerate(request.messages):
        last_user_message = i == total_messages - 1 and message.role == "user"
        content = process_content(
            message.content, message, tools, images, response_format, last_user_message
        )
        messages.append({"role": message.role, "content": content})

    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, images))

    # Prompting with images is incompatible with system messages.
    if images and messages[0]["role"] == "system":
        messages = messages[1:]
    return messages, images or None
