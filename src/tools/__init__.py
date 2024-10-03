from .get_top_hf_papers import get_top_hf_papers, get_top_hf_papers_json
from .tool_utils import ToolUtils

available_tools = [
    get_top_hf_papers_json,
]

functions = {
    "get_top_hf_papers": get_top_hf_papers,
}
