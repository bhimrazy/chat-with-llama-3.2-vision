# Adapted from https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/tool_utils.py
import ast
import json
import re
import secrets
import string
from typing import List, Union
from litserve.specs.openai import ToolCall

CUSTOM_TOOL_CALL_PATTERN = re.compile(
    r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})"
)


def is_json(s):
    try:
        parsed = json.loads(s)
        # Return True for valid objects and not for ints, strings, etc
        return isinstance(parsed, dict)
    except json.JSONDecodeError:
        return False


def generate_call_id():
    """
    Generate a unique call ID starting with 'call_' followed by exactly 9 characters in the format a-z, A-Z, 0-9.

    Returns:
        str: A unique call ID.
    """
    characters = string.ascii_letters + string.digits
    unique_id = "call_" + "".join(secrets.choice(characters) for _ in range(6))
    return unique_id


def is_valid_python_list(input_string):
    """Check if the input string is a valid Python list of function calls"""
    try:
        # Try to parse the string
        tree = ast.parse(input_string)

        # Check if it's a single expression
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return False

        # Check if the expression is a list
        expr = tree.body[0].value
        if not isinstance(expr, ast.List):
            return False

        # Check if the list is empty
        if len(expr.elts) == 0:
            return False

        # Check if all elements in the list are function calls
        for element in expr.elts:
            if not isinstance(element, ast.Call):
                return False

            # Check if the function call has a valid name
            if not isinstance(element.func, ast.Name):
                return False

            # Check if all arguments are keyword arguments
            if element.args or not all(
                isinstance(arg, ast.keyword) for arg in element.keywords
            ):
                return False

        return True

    except SyntaxError:
        # If parsing fails, it's not a valid Python expression
        return False


def parse_python_list_for_function_calls(input_string):
    """
    Parse a Python list of function calls and
    return a list of tuples containing the function name and arguments
    """
    # Parse the string into an AST
    tree = ast.parse(input_string)

    # Ensure the input is a list
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(
        tree.body[0].value, ast.List
    ):
        raise ValueError("Input must be a list of function calls")

    result = []

    # Iterate through each function call in the list
    for node in tree.body[0].value.elts:
        if isinstance(node, ast.Call):
            function_name = node.func.id if isinstance(node.func, ast.Name) else None
            function_args = {}

            # Extract keyword arguments
            for keyword in node.keywords:
                function_args[keyword.arg] = ast.literal_eval(keyword.value)

            result.append((function_name, function_args))

    return result


def prepare_tool(tool_name: str, args: dict):
    """
    Prepare a tool call in the format expected by the API.

    Args:
        tool_name (str): The name of the tool.
        args (dict): The arguments to pass to the tool
    """
    return {
        "id": generate_call_id(),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args),
        },
    }


class ToolUtils:
    @staticmethod
    def maybe_extract_custom_tool_calls(
        message_body: str,
    ) -> Union[List[ToolCall], None]:
        # {"type": "function", "name": "function_name", "parameters": {...}
        # <function=example_function_name>{"example_name": "example_value"}</function>
        # [func_name1(params_name1='params_value1', params_name2='params_value2'), func_name2(params)]

        # Extracts custom tool call from message body
        tools = []
        match = re.match(CUSTOM_TOOL_CALL_PATTERN, message_body)
        if match:
            tool_name = match.group("function_name")
            query = match.group("args")
            try:
                args = json.loads(query.replace("'", '"'))
                tool = prepare_tool(tool_name, args)
                tools.append(tool)
            except Exception as e:
                print(
                    "Exception while parsing json query for custom tool call", query, e
                )

        elif is_json(message_body):
            response = json.loads(message_body)
            if ("type" in response and response["type"] == "function") or (
                "name" in response
            ):
                function_name = response["name"]
                args = response["parameters"]
                tool = prepare_tool(function_name, args)
                tools.append(tool)

        elif is_valid_python_list(message_body):
            res = parse_python_list_for_function_calls(message_body)
            for func_name, args in res:
                tool = prepare_tool(func_name, args)
                tools.append(tool)

        return tools or None


if __name__ == "__main__":
    # example 1
    message_body = '{"type": "function", "name": "function_name", "parameters": { "example_name": "example_value"}}'
    result = ToolUtils.maybe_extract_custom_tool_calls(message_body)
    print(result)

    # example 2
    message_body = (
        '<function=example_function_name>{"example_name": "example_value"}</function>'
    )
    result = ToolUtils.maybe_extract_custom_tool_calls(message_body)
    print(result)

    # example 3
    message_body = "[func_name1(params_name1='value1', params_name2=1), func_name2(params_name1='value2', params_name2=2)]"
    result = ToolUtils.maybe_extract_custom_tool_calls(message_body)
    print(result)

    # example 4
    input_string = "[func1(arg1='value1', arg2=10)]"
    result = ToolUtils.maybe_extract_custom_tool_calls(input_string)
    print(result)
