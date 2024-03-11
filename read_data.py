#%%
import pandas as pd
import streamlit as st

def read_excel_file(file_path, sheet_name, usecols, nrows):
    return pd.read_excel(
        io=file_path,
        engine="openpyxl",
        sheet_name=sheet_name,
        skiprows=0,
        usecols=usecols,
        nrows=nrows,
    )

# Assuming Excel files are in the same directory as the script
resume_df = read_excel_file("C:/Users/supra/Desktop/ACP/Resume_test.xlsx", "Sheet1", "A:E", 8424)
job_df = read_excel_file("C:/Users/supra/Desktop/ACP/Job_data.xlsx", "Sheet1", "A:F", 8424)
job2_df = read_excel_file("C:/Users/supra/Desktop/ACP/test_job.xlsx", "Sheet1", "A:G", 8424)

# Strip unnecessary characters and clean up data
job2_df['normalizedLocations'] = job2_df['normalizedLocations'].str.strip('[]').str.replace("'", '')
job2_df['skills'] = job2_df['skills'].str.strip('[]').str.replace("'", '')
job2_df['jobTypes'] = job2_df['jobTypes'].str.strip('[]').str.replace("'", '')
job2_df['jobFunctions'] = job2_df['jobFunctions'].str.strip('[]').str.replace("'", '')

# fix resume dataset
resume_df['Experience']=resume_df['Experience'].str.extract('(\d+)')


job2_df.head(20)

resume_df.head(30)

#%%
