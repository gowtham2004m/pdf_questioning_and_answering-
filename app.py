# from flask import Flask, render_template, request
# from transformers import BertForQuestionAnswering, BertTokenizer
# import fitz  # PyMuPDF
# import torch
# from flaskext.mysql import MySQL
# import pymysql.cursors

# app = Flask(__name__)

# # MySQL configurations
# app.config['MYSQL_DATABASE_HOST'] = 'localhost'
# app.config['MYSQL_DATABASE_USER'] = 'root'
# app.config['MYSQL_DATABASE_PASSWORD'] = 'gowtham@123'
# app.config['MYSQL_DATABASE_DB'] = 'pdf_answer'

# # Initialize MySQL
# mysql = MySQL(app)

# # Load pre-trained BERT model and tokenizer
# model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
# model = BertForQuestionAnswering.from_pretrained(model_name)
# tokenizer = BertTokenizer.from_pretrained(model_name)

# # Initialize MySQL connection
# connection = pymysql.connect(host=app.config['MYSQL_DATABASE_HOST'],
#                              user=app.config['MYSQL_DATABASE_USER'],
#                              password=app.config['MYSQL_DATABASE_PASSWORD'],
#                              db=app.config['MYSQL_DATABASE_DB'],
#                              charset='utf8mb4',
#                              cursorclass=pymysql.cursors.DictCursor)


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return "No file part"

#     file = request.files['file']

#     if file.filename == '':
#         return "No selected file"

#     # Save the uploaded PDF
#     pdf_path = 'uploaded_file.pdf'
#     file.save(pdf_path)

#     return render_template('upload.html', pdf_path=pdf_path)

# @app.route('/ask_question', methods=['POST'])
# def ask_question():
#     pdf_path = request.form['pdf_path']
#     question = request.form['question']

#     # Read PDF and generate answer
#     answer = generate_answer_from_pdf(pdf_path, question)

#     return render_template('result.html', answer=answer)

# def generate_answer_from_pdf(pdf_path, question):
#     if "lines" in question.lower():
#         total_lines = read_pdf_and_count_lines(pdf_path)
#         answer = f"The PDF contains {total_lines} lines."
#     else:
#         # Read PDF
#         doc = fitz.open(pdf_path)
#         text = ""
#         for page_num in range(doc.page_count):
#             page = doc[page_num]
#             text += page.get_text()

#         # Use BERT for question answering
#         inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
#         input_ids = inputs["input_ids"]
#         token_type_ids = inputs["token_type_ids"]
     
#         # Ensure the inputs are tensors
#         input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         token_type_ids = token_type_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#         # Get model output
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)

#         answer_start_scores = outputs.start_logits
#         answer_end_scores = outputs.end_logits

#         # Find the most likely answer
#         answer_start = torch.argmax(answer_start_scores)
#         answer_end = torch.argmax(answer_end_scores) + 1
#         answer = tokenizer.decode(input_ids[0][answer_start:answer_end])

#     return answer



# def read_pdf_and_count_lines(pdf_path):
#     # Read PDF
#     doc = fitz.open(pdf_path)
#     total_lines = 0
#     for page_num in range(doc.page_count):
#         page = doc[page_num]
#         total_lines += len(page.get_text("text").splitlines())

#     return total_lines

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL
import os
import uuid
import fitz  # PyMuPDF
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'gowtham@123'
app.config['MYSQL_DB'] = 'pdf_reader'

# Initialize MySQL
mysql = MySQL(app)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded PDF
    pdf_filename = str(uuid.uuid4()) + '.pdf'  # Generating unique filename
    pdf_path = os.path.join('uploads', pdf_filename)
    file.save(pdf_path)

    return redirect(url_for('ask_question_page', pdf_path=pdf_path))

@app.route('/ask_question', methods=['POST'])
def ask_question():
    pdf_path = request.form['pdf_path']
    question = request.form['question']

    # Read PDF and generate answer
    answer = generate_answer_from_pdf(pdf_path, question)

    # Store the PDF, question, and answer in the database
    store_data_in_database(pdf_path, question, answer)

    return render_template('result.html', answer=answer)

@app.route('/ask_question/<pdf_path>', methods=['GET'])
def ask_question_page(pdf_path):
    return render_template('upload.html', pdf_path=pdf_path)


def generate_answer_from_pdf(pdf_path, question):
    try:
        if "number of lines" in question.lower():
            # Count lines
            num_lines = count_lines_in_pdf(pdf_path)
            if num_lines is None:
                return "An error occurred while counting the number of lines."
            return f"The total number of lines in the PDF is: {num_lines}"
        else:
            # Use BERT for question answering
            # Read PDF
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()

            if len(text.strip()) == 0:
                return "No text found in the PDF."

            # Use BERT for question answering
            inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            
            # Ensure the inputs are tensors
            input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            token_type_ids = token_type_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Get model output
            with torch.no_grad():
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)

            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Find the most likely answer
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = tokenizer.decode(input_ids[0][answer_start:answer_end])

            if answer.strip() == "":
                return "Model couldn't find a suitable answer to the question."

            return answer

    except Exception as e:
        print(f"Error occurred while generating answer from PDF: {e}")
        return "An error occurred while processing the PDF."


def count_lines_in_pdf(pdf_path):
    try:
        # Read PDF
        doc = fitz.open(pdf_path)
        num_lines = 0
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            num_lines += text.count('\n') + 1  # Counting newlines and adding 1 for the last line
        return num_lines

    except Exception as e:
        print(f"Error occurred while counting lines in PDF: {e}")
        return None


def store_data_in_database(pdf_path, question, answer):
    try:
        conn = mysql.connection
        cursor = conn.cursor()

        # Read PDF file data
        with open(pdf_path, 'rb') as file:
            pdf_data = file.read()

        # Insert PDF file, question, and answer into the database
        cursor.execute("INSERT INTO pdf_data (pdf_file, question, answer) VALUES (%s, %s, %s)",
                        (pdf_data, question, answer))

        # Commit your changes in the database
        conn.commit()
    except Exception as e:
        print(f"Error occurred while storing data in the database: {e}")
        # Rollback in case there is any error
        conn.rollback()
    finally:
        # Close cursor
        cursor.close()


if __name__ == '__main__':
    app.run(debug=True)
