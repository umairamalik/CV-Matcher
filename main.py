import os
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

app = Flask(__name__)

def pdf_to_text(pdf_path):
    # ... (your existing pdf_to_text function)
    try:
        # Open the PDF file in read-binary mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(file)

            # Initialize an empty string to store the text content
            text_content = ''

            # Iterate over each page of the PDF
            for page in reader.pages:
                # Extract the text from the current page
                text_content += page.extract_text()

            return text_content

    except FileNotFoundError:
        print(f"File not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
def get_best_matching_cv(description, cv_paths):
    # ... (your existing get_best_matching_cv function)
    best_cv = None
    best_similarity_score = 0

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Collect text content from all CVs
    cv_texts = []
    for pdf_path in cv_paths:
        pdf_text = pdf_to_text(pdf_path)
        if pdf_text:
            cv_texts.append(pdf_text)

    # Fit and transform the texts into TF-IDF feature vectors
    tfidf_matrix = vectorizer.fit_transform(cv_texts)

    # Calculate the similarity score between the description and each CV
    for i, pdf_text in enumerate(cv_texts):

        similarity_score = cosine_similarity(tfidf_matrix[i], vectorizer.transform([description]))[0][0]

        # Check if the current CV has a higher similarity score
        if similarity_score > best_similarity_score:
            best_similarity_score = similarity_score
            best_cv = cv_paths[i]

    return best_cv, best_similarity_score
@app.route('/', methods=['GET', 'POST'])
def index():
    best_cv = None
    best_similarity_score = 0

    if request.method == 'POST':
        description = request.form['description']
        folder_path = request.form['folder_path']

        files_in_folder = os.listdir(folder_path)
        cv_paths = [os.path.join(folder_path, file) for file in files_in_folder if file.lower().endswith('.pdf')]

        best_cv, best_similarity_score = get_best_matching_cv(description, cv_paths)

    return render_template('index.html', best_cv=best_cv, similarity_score=best_similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
