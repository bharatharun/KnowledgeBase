# app/api.py

import os
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import sqlite3
from flask import Flask, request, jsonify
import pymupdf
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import msal
import webbrowser
import http.server
import socketserver
import urllib.parse
import requests

app = Flask(__name__)



# Initialize Pinecone for vector storage and search
# pinecone = Pinecone(api_key='c957c561-ee00-4050-ad69-a65e5661460c')
# index = pinecone.Index("document-index")

CLIENT_ID = "428199b4-9643-48bb-831a-5938d39f9e05"
TENANT_ID = "consumers"
AUTHORITY = f'https://login.microsoftonline.com/{TENANT_ID}'
CLIENT_SECRET = "W_28Q~k64Fp1VRFVBxrKedPxmqQmR32htg7ZJbFJ"
REDIRECT_URI = "http://localhost:8040"
SCOPES = ['Files.Read']


OPENAI_API_KEY = "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"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

PINECONE_API_KEY = "c957c561-ee00-4050-ad69-a65e5661460c"
PINECONE_INDEX_NAME ="document-index"

# Pinecone Client initialization
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
app_msal = msal.ConfidentialClientApplication(CLIENT_ID, authority=AUTHORITY,client_credential=CLIENT_SECRET  )

UPLOAD_FOLDER = './uploads'



# Initialize SQLite Database for user management and document tracking
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS documents 
                    (id INTEGER PRIMARY KEY, 
                     filename TEXT, 
                     file_id TEXT UNIQUE, 
                     metadata TEXT, 
                     content BLOB)''')
    conn.close()

def split_document_into_chunks(file_content):
    # Initialize the RecursiveTextSplitter with desired parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=200,  # Overlap between chunks to maintain context
        length_function=len  # Function to calculate length of text
    )
    
    # Split the document content into chunks
    chunks = text_splitter.split_text(file_content)
    return chunks
# Function to store document embeddings in Pinecone
def store_embedding(text, filename):
    try:
        response = openai_client.embeddings.create(input=[text], model="text-embedding-3-large")
        embedding = response.data[0].embedding
        
        # Upsert the embedding along with metadata (filename and text)
        metadata = {
            "filename": filename,
            "text": text  
        }
        pinecone_index.upsert([(filename, embedding, metadata)])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@app.route('/test', methods=['Get'])
def test():
    return jsonify({"message": "Login Successful"})

# Admin route for uploading documents
@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        file = request.files['file']
        if file:
            
            file_content=""
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension == 'txt':
                file_content = file.read().decode('utf-8')
            elif file_extension == 'pdf':
                doc = pymupdf.open(file_path) 
                for page in doc: 
                    file_content += page.get_text("text")
                doc.close()
            elif file_extension == 'docx':
                doc = docx.Document(file_path)
                file_content = '\n'.join([para.text for para in doc.paragraphs])
            else:
                return jsonify({"error": "Unsupported file type"}), 400
            
            # Chunking the document
            chunks = split_document_into_chunks(file_content)

            filename = file.filename
            # Store each chunk in Pinecone
            for i, chunk in enumerate(chunks):
                store_embedding(chunk, f"{filename}_{i}")
            
            # Store metadata in SQLite
            conn = sqlite3.connect('database.db')
            conn.execute("INSERT INTO documents (filename, metadata) VALUES (?, ?)", (filename, ""))
            conn.commit()
            conn.close()

            return jsonify({"message": "File uploaded and processed successfully!"})
        return jsonify({"error": "No file provided!"}), 400
    except Exception as e:
        return jsonify({f"error": {e}}), 500

# User route for querying documents
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get("query")

    # Get embeddings for the user query
    query_embedding = openai_client.embeddings.create(input=[user_query], model="text-embedding-3-large")

    # Perform search in Pinecone
    search_response = pinecone_index.query(
                        vector=query_embedding.data[0].embedding,
                        top_k=3,
                        include_values=True,
                        include_metadata=True
                    )
    # Get most relevant document from search results
    most_relevant_doc = search_response['matches'][0]['metadata']['text']

    return jsonify({"response": most_relevant_doc})

# User Registration endpoint
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    # Check if the user already exists
    conn = sqlite3.connect('database.db')
    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user:
        return jsonify({"error": "User already exists!"}), 400
    else:
        # Insert new user into the users table
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully!"}), 201
    
# User login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    

    conn = sqlite3.connect('database.db')
    
    cursor = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()

    if user:
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401



@app.route('/upload_from_onedrive', methods=['GET'])
def upload_from_onedrive():
    auth_url = app_msal.get_authorization_request_url(SCOPES, redirect_uri=REDIRECT_URI)
    print(f'Please go to this URL and log in: {auth_url}')
    webbrowser.open(auth_url)
    
    
    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed_path = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed_path.query)
            if 'code' in query:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Authorization code received. You can close this window.')
                self.server.auth_code = query['code'][0]
            else:
                self.send_response(400)
                self.end_headers()
                
    
    with socketserver.TCPServer(("", 8040), MyHandler) as httpd:
        print("Listening for redirect...")
        httpd.handle_request()

   
    result = app_msal.acquire_token_by_authorization_code(httpd.auth_code, scopes=SCOPES, redirect_uri=REDIRECT_URI)

    
    if 'access_token' in result:
        headers = {'Authorization': f'Bearer {result["access_token"]}'}
        print(headers)
        upload_files_from_onedrive('root', headers)
        return jsonify({"message": "Files uploaded from OneDrive successfully!"})
    else:
        return jsonify({"error": f"Failed to acquire token: {result.get('error_description')}"}), 500

def upload_files_from_onedrive(folder_id, headers, indent=0):
    url = f'https://graph.microsoft.com/v1.0/me/drive/items/{folder_id}/children'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        items = response.json().get('value', [])
        for item in items:
            if 'folder' in item:
                upload_files_from_onedrive(item['id'], headers, indent + 1)  
            else:
                download_and_store_file(item['id'], item['name'], headers)
    else:
        print(f'Error retrieving files: {response.status_code} - {response.text}')

def download_and_store_file(file_id, filename, headers):
    download_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content'
    response = requests.get(download_url, headers=headers)

    if response.status_code == 200:
        file_content = response.content
        store_file_in_db_and_pinecone(file_id, filename, file_content)
    else:
        print(f'Error downloading file {filename}: {response.status_code} - {response.text}')

def store_file_in_db_and_pinecone(file_id, filename, content):
    try:
        # file_text = content.decode("utf-8")  # Decoding bytes to string     
        # Store in SQLite database
        # conn = sqlite3.connect('database.db')
        # conn.execute("INSERT INTO documents (filename, file_id, metadata, content) VALUES (?, ?, ?, ?)",
        #              (filename, file_id, "{}", ""))
        # conn.commit()
        # conn.close()
        
         # Determine file type
        if filename.endswith('.pdf'):
            # Extract text from PDF
            file_text = extract_text_from_pdf(content)
            
        else:
            # Convert bytes to string using UTF-8 encoding
            file_text = content.decode("utf-8")  # This will work for text files
            
            
        chunks = split_document_into_chunks(file_text)
        # Store each chunk in Pinecone
        for i, chunk in enumerate(chunks):
            embedding_response = openai_client.embeddings.create(input=[chunk], model="text-embedding-3-large")
            embedding = embedding_response.data[0].embedding
            metadata = {"filename": filename, "text": chunk,}
            pinecone_index.upsert([(file_id, embedding, metadata)])

        print(f"File '{filename}' stored in database and Pinecone successfully!")
    except sqlite3.IntegrityError:
        print(f"File '{filename}' with ID '{file_id}' already exists in the database.")
    except Exception as e:
        print(f"Error processing file '{filename}': {e}")
        
def extract_text_from_pdf(file_content):
    """Extracts text from a PDF using PyMuPDF."""
    # Load the PDF content into a PyMuPDF document
    pdf_document = pymupdf.open(stream=file_content, filetype="pdf")

    # Extract text from each page
    pdf_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text("text")  # Extract text from the page

    # Close the document after extraction
    pdf_document.close()

    return pdf_text
        
# Admin route to view uploaded documents
@app.route('/documents', methods=['GET'])
def get_documents():
    conn = sqlite3.connect('database.db')
    cursor = conn.execute("SELECT * FROM documents")
    documents = [{"id": row[0], "filename": row[1]} for row in cursor.fetchall()]
    conn.close()
    return jsonify(documents)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
