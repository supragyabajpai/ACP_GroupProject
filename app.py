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













































































































































## --------------FRONTEND------------------ ##


with st.container():
    st.subheader("Hi, Welcome to the playlist genrator..")
    st.write("This app utilizes the renowned Spotify dataset to generate personalized playlists based on your favorite song. Upon inputting a track name, the model identifies the cluster to which the track belongs and suggests a playlist of 10 songs from that cluster, offering users dynamically tailored playlists to match their preferences. Give it a try!"
    )
    st.write("Limitations: The dataset is restricted to a smaller number of clusters. To access the full range of clusters, it's recommended to download the dataset and execute the code on your local machine.")
    st.write("[Dataset](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)")
    st.write("[Check out how I am predicting the popularity of a song.](https://github.com/supragyabajpai/Playlist_Recommendation/blob/main/Code_file.ipynb)")
'''
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Look for available songs from the dataset.")

        filter_button = st.checkbox("Toggle Filters")
        filter_container = st.container()

        df_selection = df.copy()

        # If filters are toggled, show filter options
        if filter_button:
            with filter_container:
                st.write("Filters:")

                # Multiselect for genre
                genre = st.multiselect(
                    "Select the genre:",
                    options=df["genre"].unique(),
                    default=df.loc[df["genre"] == "acoustic", "genre"].unique()
                )

                # Multiselect for year
                year = st.multiselect(
                    "Select the year:",
                    options=df["year"].unique()
                )

                # Apply filters
                if genre:
                    df_selection = df_selection[df_selection["genre"].isin(genre)]
                if year:
                    df_selection = df_selection[df_selection["year"].isin(year)]

        st.write(df_selection[['artist_name', 'track_name', 'genre', 'year']], hide_index=True)



    with right_column:
        st.subheader("Enter your favourite song and get a playlist!")

        # Track recommendation
        track_name_input = st.text_input("Enter a track name:", placeholder="Yellow")

        if st.button("Get Recommendations"):
            if track_name_input:
                if track_name_input.lower() in df['track_name'].str.lower().values:
                    recommendations = get_similar_popular_tracks(track_name_input.lower(), df)
                    st.success("Recommended Tracks:")
                    
                    # Create a container to display recommended tracks as a dataframe
                    recommendations_container = st.container()

                    # Display recommended tracks dataframe
                    with recommendations_container:
                        recommendations_df = df[df['track_name'].isin(recommendations)]
                        st.dataframe(recommendations_df[['artist_name', 'track_name']], hide_index=True)
                else:
                    st.error("Track name not found in the dataset. Please make sure you are entering a valid track name.")
            else:
                st.warning("Please enter a track name.")


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

'''