import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import openai

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.response.pprint_utils import pprint_response


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# jobs 
persist_dir_j = './storageDataJobs' 
data_j  = './DataJobs'
# Resume 
persist_dir_r = './storageDataResume'
data_r = './DataResume'

storage_context_r = StorageContext.from_defaults(persist_dir=persist_dir_r)
storage_context_j = StorageContext.from_defaults(persist_dir=persist_dir_j)

index_r = load_index_from_storage(storage_context=storage_context_r)
index_j = load_index_from_storage(storage_context=storage_context_j)

retriever_r = VectorIndexRetriever(index=index_r,similarity_top_k=5)
retriever_j = VectorIndexRetriever(index=index_j,similarity_top_k=5)

postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)

query_engine_j = RetrieverQueryEngine(retriever=retriever_j,node_postprocessors=[postprocessor])
query_engine_r = RetrieverQueryEngine(retriever=retriever_r,node_postprocessors=[postprocessor])

# https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
# check the text chunck size for the query.
def query_pipeline(question, selected_jobs):
    """
    Adjust the query pipeline to limit the query's response based on selected jobs.
    
    Parameters:
    - question: The user's question.
    - selected_jobs: A list of DataFrame rows representing the selected jobs.
    """

    start = "start with : Welcome to Dotslive's consultation bot. Sure, i would like to help you..\n "
    # Assuming you modify the query_engine_j.query method to accept job filters
    selected_jobs_str = ""
    i = 1
    for job in selected_jobs:
        # Access job details using column names instead of indices
        selected_jobs_str += f"{i}) {job['Job_title']} at {job['Company']}\n"
        i += 1
    
    # Here you'll need to fix how you handle `result_str` and `selected_jobs_str`
    # Ensure you're using them correctly below; you might need a different variable name or approach
    
    response_j_1 = query_engine_j.query(selected_jobs_str)  # Modify to use selected_jobs_str or an appropriate variable
    response_j_2 = query_engine_j.query(question)  # Modify to use selected_jobs_str or an appropriate variable
    response_r = query_engine_r.query(question)
    prompt_j = " content from jobs: "
    prompt_r = "content from Resume:"
    
    # Ensure the concatenation below makes sense and uses the correct variables
    return start + prompt_r + response_r.response + prompt_j + response_j_1.response + response_j_2.response




# Function to clean HTML content and extract text

def chat_with_bot(user_question, selected_jobs):


    # Query the OpenAI API with the prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an experienced and deligent career consultant. If the question is about a job or career advice, You respond with very short answers and do not entertain answering to other questions."
            },
            {
                "role": "user",
                "content": query_pipeline(user_question, selected_jobs)
            }
        ],
        temperature=0.4,
        max_tokens=512,
        stop=None,
    )
    return response.choices[0].message.content

def display_resume(resume):
    st.subheader("Resume")
    raw_html = resume.iloc[0]['html']
    st.write(raw_html, unsafe_allow_html=True)


def display_job_html(raw_html):
    st.markdown(
        f'<a href="{raw_html}" target="_blank">Open Job Page</a>',
        unsafe_allow_html=True
    )


def main():
    st.title("Career Adviser - powered by GPT")
    resume = pd.read_csv('./DataResume/ResumeDisplay.csv')
    jobs = pd.read_csv('./DataJobs/JobsDisplay.csv')

    st.sidebar.subheader("Recent Jobs Visited")

    # Initialize a list to keep track of selected jobs based on toggle state
    selected_jobs = []

    # Create a toggle switch for each job
    for i, job in jobs.iterrows():
        # Use a unique key for each toggle to maintain state independently
        toggle = st.sidebar.toggle(f"{job['Job_title']} - {job['Company']}", key=f"toggle_{i}")
        if toggle:
            # If the toggle is 'on', add the job to the list of selected jobs
            selected_jobs.append(job)


    if selected_jobs:
        st.subheader("Selected Jobs")
        # Display details for each selected job with URLs
        for i, job in enumerate(selected_jobs, 1):
            # Generate URL for each job
            job_url = job['url']
            st.write(f"{i}) [{job['Job_title']} at {job['Company']}](<{job_url}>)")
        
        user_question = st.text_input("Type your question here:")
    
        if user_question:

            response = chat_with_bot(user_question, selected_jobs)  
            st.text_area("Response", value=response, disabled=True)
    
    else:
        display_resume(resume)  




if __name__ == "__main__":
    main()
