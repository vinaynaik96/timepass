import streamlit as st
import tempfile
import os

# Existing code ...

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = conversation({"question": prompt})
    with st.chat_message("assistant"):
        st.markdown(response["text"])
    st.session_state.messages.append({"role": "assistant", "content": response["text"]})

    # Creating a temporary file to store Python code
    temp_filename = tempfile.NamedTemporaryFile(suffix=".py", delete=False).name
    with open(temp_filename, "w") as temp_file:
        temp_file.write(response["text"])  # Writing the code content to the temp file

    # Provide the download button with the file content
    with open(temp_filename, "r") as file:
        code_content = file.read()
        st.download_button(
            label='Download .py File',
            data=code_content.encode(),
            file_name='generated_code.py',
            mime='text/plain'
        )

    os.remove(temp_filename)  # Clean up: Remove the temporary file

    st.success("Code downloaded successfully!")
