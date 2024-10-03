import json

import streamlit as st

from src.config import IMAGE_EXTENSIONS, SYSTEM_MESSAGE
from src.ui.utils import all_images, encode_image
from src.tools import available_tools


def file_upload():
    # Sidebar header
    st.sidebar.header("Upload Files")
    uploaded_files, file_objects = None, None

    # File uploader with improved grammar and standardized code
    uploaded_files = st.sidebar.file_uploader(
        "Please select either up to 3 images or a single video file...",
        type=IMAGE_EXTENSIONS,
        accept_multiple_files=True,
    )

    if uploaded_files is not None and len(uploaded_files) > 0:
        # check if all of the uploaded files are images or videos
        if all_images(uploaded_files) and len(uploaded_files) <= 3:
            with st.sidebar.status("Processing image..."):
                file_objects = [encode_image(image) for image in uploaded_files]
                st.sidebar.image(uploaded_files, use_column_width=True)
        else:
            st.error(
                "Please upload max up to 3 images file with the following extensions: "
                + ", ".join(IMAGE_EXTENSIONS)
            )

    return uploaded_files, file_objects


def header():
    # CSS to center crop the image in a circle
    circle_image_css = """
    <style>
    .center-cropped {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 50%;
        width: 96px;
        height: 96px;
        object-fit: cover;
    }
    </style>
    """

    # Inject CSS
    st.markdown(circle_image_css, unsafe_allow_html=True)

    st.markdown(
        """
        <img src="https://github.com/user-attachments/assets/645d4447-eb8a-4992-9c53-8c37e904e82f" class="center-cropped">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align: center; font-size:1.5rem'>Chat with Llama 3.2-Vision (11B) multimodal LLM</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align: center; margin-bottom:4'>"
        "<p style='font-size:0.9rem'>The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. <a href='https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct' target='_blank'>Read more</a></p>"
        "</div>",
        unsafe_allow_html=True,
    )


def system_prompt():
    st.sidebar.header("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Modify the prompt here.", value=SYSTEM_MESSAGE["content"]
    )
    SYSTEM_MESSAGE["content"] = system_prompt
    return system_prompt


# Advanced settings
def validate_json(input_text):
    try:
        json.loads(input_text)
    except json.JSONDecodeError:
        st.sidebar.warning("Invalid JSON format. Please correct the input.")
        pass


def advanced_settings():
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Settings")
    tools, schema = None, None

    enable_tools = st.sidebar.toggle("Tools")
    if enable_tools:
        selected_tools = [
            tool["function"]["name"]
            for tool in available_tools
            if st.sidebar.checkbox(tool["function"]["name"], value=True)
        ]

        # Filter available tools based on selected tools
        tools = [
            tool
            for tool in available_tools
            if tool["function"]["name"] in selected_tools
        ]

    json_mode = st.sidebar.toggle("JSON mode")
    if json_mode:
        st.sidebar.markdown(
            "[Guide on Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction)"
        )

        # Store the initial schema in session state if not already stored
        if "schema" not in st.session_state:
            st.session_state.schema = '{"type": "json_object"}'

        def reformat_json():
            try:
                json_str = st.session_state.schema
                json_object = json.loads(json_str)
                formatted_json = json.dumps(json_object, indent=2)
                st.session_state.schema = formatted_json
            except json.JSONDecodeError:
                pass

        schema = st.sidebar.text_area(
            "Edit Schema",
            value=st.session_state.schema,
            help="Define the schema for structured output",
            on_change=reformat_json,
        )
        validate_json(schema)
        schema = json.loads(schema)

    return tools, schema
