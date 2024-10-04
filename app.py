import json

import streamlit as st

from src.api import client
from src.config import MODEL, SYSTEM_MESSAGE
from src.tools import functions
from src.ui.components import advanced_settings, file_upload, header, system_prompt
from src.ui.utils import all_images, prepare_content_with_images


def main():
    # Title section
    header()

    # Add input field for system prompt
    system_prompt()

    # Sidebar section for file upload
    uploaded_files, file_objects = file_upload()

    # Advanced Settings
    tools, response_format = advanced_settings()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    if "messages" in st.session_state.keys() and len(st.session_state.messages) > 0:
        # add clear chat history button to sidebar
        st.sidebar.button(
            "Clear Chat History",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Display chat message in chat message container
        role = message["role"]
        if role in ["user", "assistant"]:
            with st.chat_message(role):
                content = message["content"]
                if isinstance(content, list):
                    urls = [
                        item["image_url"]["url"]
                        for item in content
                        if "image_url" in item
                    ]
                    text = [item["text"] for item in content if "text" in item]
                    st.markdown("\n".join(text))
                    st.image(urls if len(urls) < 3 else urls[0], width=200)
                else:
                    st.markdown(content)

    if prompt := st.chat_input("Ask something", key="prompt"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            content = (
                prepare_content_with_images(prompt, file_objects)
                if file_objects
                else prompt
            )
            if file_objects:
                if all_images(uploaded_files):
                    st.image(uploaded_files, width=200)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": content})

        # Get response from the assistant
        with st.chat_message("assistant"):
            messages = [SYSTEM_MESSAGE, *st.session_state.messages]

            if not tools:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                    response_format=response_format,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            else:
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls
                    if tool_calls:
                        with st.status("Thinking...", expanded=True) as status:
                            for tool_call in tool_calls:
                                function_name = tool_call.function.name
                                tool = functions[function_name]
                                args = json.loads(tool_call.function.arguments)
                                st.write(
                                    f"Calling {function_name}... with args: {args}"
                                )
                                tool_response = tool(**args)
                                st.write(f"Tool Response: {tool_response}")
                                st.session_state.messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "ipython",
                                        "content": tool_response,
                                        "name": function_name,
                                    }
                                )
                                status.update(
                                    label=f"Running {function_name}... Done!",
                                    state="complete",
                                    expanded=False,
                                )
                        stream = client.chat.completions.create(
                            model=MODEL, messages=st.session_state.messages, stream=True
                        )
                        response = st.write_stream(stream)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        st.write(response_message.content)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_message.content}
                        )


if __name__ == "__main__":
    main()
