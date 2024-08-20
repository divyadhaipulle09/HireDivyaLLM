import PyPDF2
import json

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text_data = {}
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text_data[f'Page_{i+1}'] = page.extract_text()
    return text_data

# Function to convert text data to JSON
def convert_pdf_to_json(pdf_file, json_file):
    text_data = extract_text_from_pdf(pdf_file)
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(text_data, file, indent=4, ensure_ascii=False)
    print(f"PDF content has been converted to JSON and saved as {json_file}")

# Specify the path to your PDF file and the output JSON file
pdf_file = 'HireDivyaResume.pdf'
json_file = 'statements.json'

# Convert PDF to JSON
convert_pdf_to_json(pdf_file, json_file)
