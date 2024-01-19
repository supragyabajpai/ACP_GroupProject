import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Resume and Job data
resume = pd.read_csv(r"C:\Users\srika\OneDrive\Documents\York\Sem-2 york\MBAN 6090 - Analytics Consulting Project\webscrape\communitech\resume_data_231230.csv")
jobs = pd.read_csv(r"C:\Users\srika\OneDrive\Documents\York\Sem-2 york\MBAN 6090 - Analytics Consulting Project\webscrape\communitech\jobs_info.csv")


# Backup original data
resume['Resume_i'] = resume['Resume']
jobs['description_i'] = jobs['description']

# Text Processing Functions
def text_preprocessing(text):
    # Convert text to lowercase
    text = str(text).lower()
    # Remove newline characters
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Stem 
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in string.punctuation]
    # Join tokens back into text
    return ' '.join(tokens)

# Apply Text Preprocessing
resume['Resume'] = resume['Resume'].apply(text_preprocessing)
jobs['description'] = jobs['description'].fillna('').apply(text_preprocessing)

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
all_texts = pd.concat([resume['Resume'], jobs['description']])
vectorizer.fit(all_texts)

# Transform resumes and job postings
resumes_tfidf = vectorizer.transform(resume['Resume'])
job_postings_tfidf = vectorizer.transform(jobs['description'])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(resumes_tfidf, job_postings_tfidf)

# Initialize Flask app
app = Flask(__name__)

# Function to find matching jobs
# Function to find matching jobs with descending scores
def find_matching_jobs(user_id, page=0, items_per_page=10):
    job_matches = similarity_matrix[user_id]
    # Sort indices based on similarity scores in descending order
    sorted_indices = np.argsort(job_matches)[::-1]  
    start = page * items_per_page
    end = start + items_per_page
    top_matches_indices = sorted_indices[start:end]
    
    # Directly map similarity scores to percentage
    normalized_scores = [round(100 if (score * 100 * 4) > 100 else (score * 100 * 4), 2) for score in job_matches[top_matches_indices]]

    return list(zip(top_matches_indices, normalized_scores))


# Function to display resumes and job matches
def display_resumes_jobs(user_id, page=0):
    # Display resume details
    html_output = '<h1>Resumes</h1><div style="display:flex;">'
    resume_details = resume.iloc[user_id]['Raw_html']  
    html_output += f'<div style="flex:1; padding:10px;">{resume_details}</div>'
    html_output += '<div style="flex:1; padding:10px;"><ol>'
    # Display job matches
    matching_jobs = find_matching_jobs(user_id, page)
    for job_id, score in matching_jobs:
        job_title = jobs.iloc[job_id]['title']
        first_line_description = jobs.iloc[job_id]['description_i'].split('.')[0][:50] + '...'        
        html_output += f'<li><a href="/job/{job_id}">{job_title} (Match: {score}%)</a></li>'

    # Update pagination links
    html_output += '</ol></div></div>'
    total_pages = -(-len(similarity_matrix[user_id]) // 10)
    html_output += '<div>Pages: ' + ' '.join([f'<form method="post" style="display:inline;"><input type="hidden" name="resume_id" value="{user_id}"><input type="hidden" name="page" value="{i}"><input type="submit" value="{i+1}"></form>' for i in range(total_pages)]) + '</div>'
    return render_template_string(html_output)

# Route for selecting a resume and displaying job matches
@app.route('/', methods=['GET', 'POST'])
# Function to select a resume and display job matches
def select_resume():
    # Display resumes and job matches
    if request.method == 'POST':
        # Get the selected resume ID
        user_id = int(request.form['resume_id'])
        page = int(request.form.get('page', 0))
        return display_resumes_jobs(user_id, page)
    else:
        # Create dropdown menu for resume selection
        options = ''.join([f'<option value="{i}">Resume {i}</option>' for i in range(len(resume))])
        return f'''
            <form method="post">
                Select Resume: <select name="resume_id">{options}</select><br>
                <input type="hidden" name="page" value="0">
                <input type="submit" value="Select">
            </form>
        '''

# Route for displaying full job description
@app.route('/job/<int:job_id>')
# Function to display full job description
def job_description_page(job_id):
    # Display full job description
    job_description = jobs.iloc[job_id]['description_i']
    return f'<h1>Job Description</h1><p>{job_description}</p>'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)