import os
import json
import time
import streamlit as st
from openai import OpenAI

OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

def create_assistant(client):
    assistant_file_path = 'assistant.json'

    # If there is an assistant.json file already, then load that assistant
    if os.path.exists(assistant_file_path):
        with open(assistant_file_path, 'r') as file:
            assistant_data = json.load(file)
            assistant_id = assistant_data['assistant_id']
            st.info("Loaded existing assistant ID.")
    else:
        st.info("Creating assistant")
        # If no assistant.json is present, create a new assistant using the below specifications
        file = client.files.create(file=open("knowledge.csv", "rb"),
                                   purpose='assistants')

        assistant = client.beta.assistants.create(
            # Getting assistant prompt from "prompts.py" file, edit on left panel if you want to change the prompt
            instructions="""Read the pdf and analyse its context for creation of an in detail summary of all unique topics and events/stories discussed in the text.""",
            model="gpt-4-1106-preview",
            tools=[
                {
                    "type": "code_interpreter"  # This adds the knowledge base as a tool
                },
            ],
            file_ids=[file.id])
        st.info(assistant.id)
        # Create a new assistant.json file to load on future runs
        with open(assistant_file_path, 'w') as file:
            json.dump({'assistant_id': assistant.id}, file)
            st.info("Created a new assistant and saved the ID.")

        assistant_id = assistant.id

    return assistant_id

def chat_request(prompt: str):
    start = time.time()
    client = OpenAI(api_key=OPENAI_API_KEY)
    assistant_id = 'xxxxxxxxxxxxxxxxxxxxxx'
    st.info("Starting a new conversation...")
    thread = client.beta.threads.create()
    st.info(f"New thread created with ID: {thread.id}")
    thread_id = thread.id

    client.beta.threads.messages.create(thread_id=thread_id,
                                        role="user",
                                        content=prompt)

    # Run the Assistant
    run = client.beta.threads.runs.create(thread_id=thread_id,
                                          assistant_id=assistant_id)
    # logging.info("Ran the Assistant")
    # Check if the Run requires action (function call)
    while True:
        # logging.info("While Loop")
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id,
                                                       run_id=run.id)
        st.info(f"Run status: {run_status.status}")
        if run_status.status == 'completed':
            # Retrieve and return the latest message from the assistant
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            response = messages.data[0].content[0].text.value

            st.info(f"Assistant response: {response}")
            break
        else:
            # logging.info("Sleep 5 seconds")
            time.sleep(5)  # Wait for a second before checking again

    p_time = time.time() - start
    st.info(f"Time take in seconds: {(p_time/60):.2f}s")
    return {"data": response, "status": "success"}
