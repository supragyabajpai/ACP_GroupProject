import pandas as pd
from datetime import datetime

def get_location_match(resume_location, job_desc_location, gmaps):
   resume_geocode = gmaps.geocode(resume_location.lower())
   job_desc_geocode = gmaps.geocode(job_desc_location.lower())

   resume_lat_lng = (resume_geocode[0]['geometry']['location']['lat'], resume_geocode[0]['geometry']['location']['lng'])
   job_desc_lat_lng = (job_desc_geocode[0]['geometry']['location']['lat'], job_desc_geocode[0]['geometry']['location']['lng'])

   distance = gmaps.distance_matrix(resume_lat_lng, job_desc_lat_lng)['rows'][0]['elements'][0]['distance']['value']

   threshold_distance = 10000

   return distance <= threshold_distance

def compare_skills(resume_skills, job_desc_skills):
   common_skills = resume_skills.intersection(job_desc_skills)

   return len(common_skills)

def compare_education(resume_education, job_desc_education):
   if pd.isna(resume_education) or pd.isna(job_desc_education):
       return 0
   elif job_desc_education == resume_education:
       return 1
   elif job_desc_education == 'masters' and resume_education == 'bachelors':
       return 0.5
   else:
       return 0

def compare_experience(resume_exp, job_desc_exp):
   return abs(resume_exp - job_desc_exp)

def compare_resumes_and_job_desc(resume_df, job_desc_df, gmaps):
   location_match = get_location_match(resume_df['Location'].str.lower(), job_desc_df['Location'].str.lower(), gmaps)

   resume_skills = set(resume_df['Skills'].str.lower().str.split(',').explode().str.strip())
   job_desc_skills = set(job_desc_df['Skills'].str.lower().str.split(',').explode().str.strip())
   matching_skills_count = compare_skills(resume_skills, job_desc_skills)

   education_match = []
   for resume_education, job_desc_education in zip(resume_df['Education'].str.lower(), job_desc_df['Education'].str.lower()):
       education_match.append(compare_education(resume_education, job_desc_education))

   resume_exp = resume_df['Experience'].astype(float)
   job_desc_exp = job_desc_df['Experience'].astype(float)
   exp_diff = list(map(compare_experience, resume_exp, job_desc_exp))


   return {
       'Location Match': location_match,
       'Matching Skills Count': matching_skills_count,
       'Education Match': education_match,
       'Experience Difference': exp_diff
   }


resume_df = pd.DataFrame("")
job_desc_df = pd.DataFrame("")

