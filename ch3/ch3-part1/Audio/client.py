import requests
import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, bytes):
            st.audio(content)
        else:
            st.markdown(content)


if prompt := st.chat_input("Write your prompt in this input field"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = requests.get(
        "http://localhost:8000/generate/audio", params={"prompt": prompt}
    )
    response.raise_for_status()
    audio_bytes = response.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": audio_bytes,
        }
    )
    with st.chat_message("assistant"):
        st.text("Here is your generated audio")
        st.audio(audio_bytes)
