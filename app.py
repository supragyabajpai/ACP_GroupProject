#%%
import pandas as pd  
import streamlit as st 
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Job Recommender", layout="wide",page_icon=":necktie:")


# Read the data from the Excel file

##-------------- BACKEND------------------ ##
@st.cache_data
def read_df():

    def read_excel_file(file_path, sheet_name, usecols, nrows):
        return pd.read_excel(
            io=file_path,
            engine="openpyxl",
            sheet_name=sheet_name,
            skiprows=0,
            usecols=usecols,
            nrows=nrows,
        )
    resume_df = read_excel_file("C:/Users/supra/Desktop/ACP/Resume_test.xlsx", "Sheet1", "A:E", 8000)
    job_df = read_excel_file("C:/Users/supra/Desktop/ACP/Job_data.xlsx", "Sheet1", "A:F", 8000)
    job2_df = read_excel_file("C:/Users/supra/Desktop/ACP/test_job.xlsx", "Sheet1", "A:G", 8000)


    # Stripping unnecessary characters and clean up data
    job2_df['normalizedLocations'] = job2_df['normalizedLocations'].str.strip('[]').str.replace("'", '')
    job2_df['skills'] = job2_df['skills'].str.strip('[]').str.replace("'", '')
    job2_df['jobTypes'] = job2_df['jobTypes'].str.strip('[]').str.replace("'", '')
    job2_df['jobFunctions'] = job2_df['jobFunctions'].str.strip('[]').str.replace("'", '')
    job2_df['minYearsExp']= job2_df['minYearsExp'].astype(float)

    return resume_df, job_df, job2_df

r_df,j1_df,j2_df = read_df()

r_df.info()
j2_df.info()

#%%
def compare_data(r_df, j2_df):
    matches = []  
    
    for index_resume, row_resume in r_df.iterrows():
        candidate_name = row_resume['Name']
        resume_skills = set(row_resume['Skills'].lower().split(', '))
        candidate_location = set(row_resume['Location'].lower().split(', '))
        candidate_experience = row_resume['Experience']
        
        for index_job, row_job in j2_df.iterrows():
            count = 0  
            loc_count = 0
            exp_match = 0

            company_name = row_job['companyName'].lower()
            job_title = row_job['title'].lower()
            jd_skills = set(row_job['skills'].lower())
            job_locations = set(row_job['normalizedLocations'].lower())
            job_experience = row_job['minYearsExp']
            
            for skill in resume_skills:
                if skill in jd_skills:
                    count += 1

            for location in candidate_location: ##  this can be improved with a better logic
                if location in job_locations:
                    loc_count += 1
            
            exp_match = candidate_experience - job_experience

            # either by cosine similarity or using openai's GPT-3 to match the job description and resume description
            
            matches.append({
                'candidate_name': candidate_name,
                'company_name': company_name,
                'job_name': job_title,
                'skills_matching': count,
                'location_match': loc_count,
                'experience_match': exp_match
            })

    
    matches_df = pd.DataFrame(matches)
    return matches_df

output = compare_data(r_df, j2_df)

#%%












































































































































