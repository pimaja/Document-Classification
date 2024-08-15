import os
import sys
import time
import argparse
import pathlib
import string
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, required=True, help="Text file which will be used a document for classification. This is a required argument.")
    args = parser.parse_args()
    return args


def open_file(args, current_path):
    # check if the argument file is provided
    if args.file is not None:
        file = args.file
    else:
        print("Error: No file provided.")
        sys.exit()

    # check if the file is provided as the filepath
    if os.path.exists(file):
        filepath = file
    # check if the file was provided from the current directory
    else:
        filepath = str(current_path) + "/" + file
        if not os.path.exists(filepath):
            print("Error: File does not exist. Please provide a file from the current directory or provide a right path to the file.")
            sys.exit()

    # open the file and check if it is a text file
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            document = file.read()
    except:
        print("Error: Cannot read the file. The file is not a text file.")
        sys.exit()

    # if the file is empty, there is nothing to classify
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        print("Error: Provided document is empty. It cannot be classified. Please provide a new document.")
        sys.exit()
    else:
        print("Opened the document")
    return document


def clean_document(document):
    print("Cleaning the document... ")
    # remove newlines
    document = document.replace('\n', '')
    # lowercase
    document = document.lower()
    # remove special characters
    punctuation_pattern = r'[^\w\s]'
    document = re.sub(punctuation_pattern, '', document)
    # remove stopwords
    new_words = word_tokenize(document)
    new_filtered_words = [w for w in new_words if w not in stopwords.words('english')]
    document = ' '.join(new_filtered_words)
    # remove single letters
    document = ' '.join( [w for w in document.split() if len(w)>1] )
    # remove urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    document = url_pattern.sub(r'', document)
    # remove html tags
    html_tags_pattern = r'<.*?>'
    document = re.sub(html_tags_pattern, '', document)
    # if the document is empty after cleaning, it is not sufficient and there is nothing to classify
    if document == "":
        print("Error: Provided document doesn't have sufficient information. It cannot be classified. Please provide a new document.")
        sys.exit()
    else:
        print(f"Cleaned document: {document} \n")
    return document


def main():
    # Timer - important for production
    start_time = time.time()

    # Parse the script argument and preprocess the document
    args = parse_args()
    current_path = pathlib.Path(__file__).parent.resolve()
    document = open_file(args, current_path)
    cleaned_document = clean_document(document)

    # Inferencing using a fine-tuned Mini-BERT and transformers from HuggingFace -> fine-tuning is done in the development.ipynb
    print("Classifying... \n")
    classifier = pipeline("text-classification", model=f"{current_path}/checkpoint")
    prediction = classifier(cleaned_document)

    # Calculate time needed for preprocessing and classifing
    elapsed_time = time.time() - start_time
    print(f"Time for preprocessing and classifing the document: {elapsed_time:.2f}s \n")

    # if the probability of a predicted class is less than 0.5, we cannot be sure it is actually in that class, it could be some text that doesn't fit in any class
    if prediction[0]['score'] < 0.5:
        print(f"Error: Prediction score is {prediction[0]['score']:.2f} for class {prediction[0]['label']} which is not enough to confidently determine that this is the right class. \n")
        sys.exit()
    else:
        print(f"Predicted class is: {prediction[0]['label']}")
        print(f"Prediction score is: {prediction[0]['score']:.2f} \n")
        return


if __name__ == "__main__":
    main()
