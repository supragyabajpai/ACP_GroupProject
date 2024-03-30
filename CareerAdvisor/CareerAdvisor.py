import streamlit as st
import pandas as pd
import os
import openai
import time
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from openai import OpenAIError

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables for assistant and thread IDs
assistant_id = None
thread_id = None
# 
# Directories for data storage
persist_dir_r = './storageDataResume'
persist_dir_j = './storageDataJobs'

# Create storage contexts
storage_context_r = StorageContext.from_defaults(persist_dir=persist_dir_r)
storage_context_j = StorageContext.from_defaults(persist_dir=persist_dir_j)

# Load indices from storage
index_r = load_index_from_storage(storage_context=storage_context_r)
index_j = load_index_from_storage(storage_context=storage_context_j)

# Create retrievers
retriever_r = VectorIndexRetriever(index=index_r, similarity_top_k=5)
retriever_j = VectorIndexRetriever(index=index_j, similarity_top_k=5)

# Create postprocessor
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)

# Create query engines
query_engine_r = RetrieverQueryEngine(retriever=retriever_r, node_postprocessors=[postprocessor])
query_engine_j = RetrieverQueryEngine(retriever=retriever_j, node_postprocessors=[postprocessor])

# Start time for response calculation
time_start = time.time()

# Memory for storing conversation history
memory = []

# Function to initialize the assistant
def initialize_assistant():
    global assistant_id
    try:
        assistant = client.beta.assistants.create(
            name="Career Advisor",
            instructions="You are an experienced career advisor. Write responses to career-related questions.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-turbo-preview",
        )
        assistant_id = assistant.id
    except OpenAIError as e:
        st.error(f"Failed to initialize assistant: {e}")

# Function to create a new thread for conversation
def create_thread():
    global thread_id
    try:
        thread = client.beta.threads.create()
        thread_id = thread.id
    except OpenAIError as e:
        st.error(f"Failed to create thread: {e}")

# Function to add a user message to the conversation thread
def add_message_to_thread(user_question):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_question
        )
    except OpenAIError as e:
        st.error(f"Failed to add message to thread: {e}")

# Function to check if user input needs moderation
def check_for_moderation(text):
    try:
        response = client.moderations.create(input=text)
        output = response.results[0]
        categories_dict = output.categories if isinstance(output.categories, dict) else vars(output.categories)
        flagged = output.flagged
        return flagged, categories_dict
    except OpenAIError as e:
        st.error(f"Moderation check failed: {e}")
        return False, {}

# Function to generate response from the assistant
def generate_response():
    class EventHandler(openai.AssistantEventHandler):
        def on_text_delta(self, delta, snapshot):
            st.write(delta.value, end="", flush=True)
    
    try:
        with client.beta.threads.runs.create_and_stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
    except OpenAIError as e:
        st.error(f"Failed to generate response: {e}")

# Function to handle the query pipeline
def query_pipeline(question, selected_jobs):
    start = "Start with : Welcome to Dotslive's consultation bot. I'll be happy to help you. After reviewing yoru resume and job description, \n"
    selected_jobs_str = "\n".join([f"{i}) {job['Job_title']} at {job['Company']}" for i, job in enumerate(selected_jobs, 1)])
    response_r = query_engine_r.query(question).response
    response_j = query_engine_j.query(question).response
    return f"{start}\n My Resume:\n{response_r}\nSelected jobs:\n{selected_jobs_str}\nContent from Jobs:\n{response_j}"

# Function to chat with the assistant
def chat_with_bot(user_question, selected_jobs):
    flagged, categories = check_for_moderation(user_question)
    if flagged:
        flagged_categories = [cat for cat, is_flagged in categories.items() if is_flagged]
        flagged_message = f"Your message was flagged for the following reason(s): {', '.join(flagged_categories)}. Please revise your input."
        return flagged_message
    
    system_prompt = "You are an experienced and diligent career consultant. If the question is about a job or career advice, respond with very precise and detailed short answers and do not entertain answering to other questions."
    user_query = query_pipeline(user_question, selected_jobs)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.4,
        max_tokens=512,
        stop=None,
    )
    
    response_content = response.choices[0].message.content
    return response_content

# Main function to run the Streamlit app
def main():
    global assistant_id, thread_id
    if not assistant_id:
        initialize_assistant()

    st.title("Career Adviser - powered by GPT :rocket:")
    resume = pd.read_csv('./DataResume/ResumeDisplay.csv')
    jobs = pd.read_csv('./DataJobs/JobsDisplay.csv')
    
    st.sidebar.subheader("Recent Jobs Visited")
    selected_jobs = []

    for i, job in jobs.iterrows():
        toggle = st.sidebar.checkbox(f"{job['Job_title']} - {job['Company']}", key=f"toggle_{i}")
        if toggle:
            selected_jobs.append(job)

    if selected_jobs:
        if not thread_id:
            create_thread()

        st.subheader("Selected Jobs :white_check_mark:")
        for i, job in enumerate(selected_jobs, 1):
            job_url = job['url']
            st.write(f"{i}) [{job['Job_title']} at {job['Company']}](<{job_url}>)")

        user_question = st.text_input("Type your question here:")
        if user_question:
            add_message_to_thread(user_question)
            response = chat_with_bot(user_question, selected_jobs)
            st.write(response)
            st.write("Time for response: {:.2f} seconds".format(time.time() - time_start))
    else:
        raw_html = resume.iloc[0]['html']
        st.subheader("Resume")
        st.write(raw_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
