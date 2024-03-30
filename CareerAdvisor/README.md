**CareerAdvisor - README**

**Overview**

CareerAdvisor is a career consultation application that leverages GPT models to provide personalized career guidance. This application integrates with Streamlit for a user-friendly UI, pandas for data handling, and custom modules for indexing and querying vector space models.

**Installation**

To install CareerAdvisor, follow these steps:

* Clone the repository:
```bash
git clone [https://github.com/supragyabajpai/ACP_GroupProject/tree/main/CareerAdvisor]
```

* Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**File Structure**

* **CareerAdvisor.py:** Main application script integrating all components.
* **DataJobs/, DataResume/:** Directories containing job listings and resumes respectively.
* **storageDataJobs/, storageDataResume/:** Directories for persistent storage of vector indices.
* **raw_data/:** Contains the cleaned datasets for jobs and resumes.
* **requirements.txt:** Lists all the dependencies for the application.
* **venv/:** Virtual environment directory.

**Usage**

After installation, run CareerAdvisor.py with Streamlit:
```bash
streamlit run CareerAdvisor.py
```

**Features**

* Data storage and retrieval using vector space models.
* Interactive querying through Streamlit UI.
* Real-time career advice using OpenAI's GPT models.

**Notebooks for Evaluation**

* **240315_part_A.ipynb:** Evaluation of initial part of the project- Preporcessing, vecotrs creation
* **240315_part_B.ipynb:** Continuation of evaluation - gradio for a short query interface for tuning the query
* **pinecone_query_verify.ipynb:** Verification of Pinecone server setup.

**Documentation and Logic**

For detailed instructions and more information, refer to the specific files and notebooks included in the project repository.
