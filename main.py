import streamlit as st
import os
from glob import glob
from pathlib import Path
import logging
try:
    from .CSV_Rag import csv_rag
    from .PDF_Rag import pdf_rag
    from .EXCEL_Rag import excel_rag
except ImportError:
    from CSV_Rag import csv_rag
    from PDF_Rag import pdf_rag
    from EXCEL_Rag import excel_rag

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
st.set_page_config(
    page_title="Multimodal RAG Q&A",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 Multimodal RAG Q&A")
st.write("Ask questions about the module/book and get detailed answers!")

# File upload widget with additional styling
st.markdown(
    """
    <style>
    .stFileUpload {
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .get-answer-button {
        display: inline-block;
        background-color: #ADD8E6; /* light blue */
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
    }
    .get-answer-button:hover {
        background-color: #5b96e3; /* darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf","csv","xlsx"], help="Upload a PDF or CSV document to start the analysis")

PDF_File = False
CSV_File = False
Excel_File = False

# To check the file type
def file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if file_extension == '.pdf':
        return 'pdf'
    elif file_extension == '.csv':
        return 'csv'
    elif file_extension == '.xlsx':
        return 'xlsx'
    else:
        return 'Unknown'

if uploaded_file is not None:
    file_type_result = file_type(uploaded_file.name)
    if file_type_result == "pdf":
        # Save uploaded file
        file_path = Path(f"Data/PDF/{uploaded_file.name}")
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        
        PDF_File = True
        CSV_File = False
        Excel_File = False
        st.success(f"Your PDF file - '{uploaded_file.name}' has been uploaded successfully!")

    elif file_type_result == "csv":
        # Save uploaded file
        file_path = Path(f"Data/CSV/{uploaded_file.name}")
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        
        PDF_File = False
        CSV_File = True
        Excel_File = False
        st.success(f"Your CSV file - '{uploaded_file.name}' has been uploaded successfully!")
    elif file_type_result == "xlsx":
        # Save uploaded file
        file_path = Path(f"Data/Excel/{uploaded_file.name}")
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        
        PDF_File = False
        CSV_File = False
        Excel_File = True
        st.success(f"Your Excel file - '{uploaded_file.name}' has been uploaded successfully!")
    else:
        st.error(f"Your file - '{uploaded_file.name}' is an unknown file type!!!")

query = st.text_input("Your question:", help="Type your question related to the uploaded document here")

if st.button("Get Answer"):
    if uploaded_file is not None and query:
        with st.spinner("Processing your question..."):
            if PDF_File:
                answer = pdf_rag(query, uploaded_file.name)
            elif CSV_File:
                answer = csv_rag(query, uploaded_file.name)
            elif Excel_File:
                answer = excel_rag(query,uploaded_file.name)
            else:
                answer = "File type not supported."

            st.write("### You:")
            st.write(query)
            st.write("### Response:")
            st.write(answer)
            
            # Displaying Images
            try:
                # Use glob to get a list of image file paths from the directory
                image_paths = glob("exports/charts/*")
                
                for image_path in image_paths:
                    st.image(image_path, caption=os.path.basename(image_path), use_column_width=True)
            except Exception as e:
                pass
            
            # Removing the image
            try:
                # Use glob to get a list of file paths in the directory
                files_to_remove = glob("exports/charts/*")
                logger.info(f"Files to remove: {files_to_remove}")

                # Remove each file
                for file_path in files_to_remove:
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing files: {e}")

    else:
        st.error("Please upload a file and enter a question before getting an answer.")
