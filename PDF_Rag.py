import os
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

# Ensure data directory exists
os.makedirs("Data/PDF", exist_ok=True)

# Setting embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Get PDF elements
def get_pdf_element(file_path):
    # To store images from pdf
    output_path = f"./Images/{file_path}"
    
    raw_pdf_elements = partition_pdf(
        filename= f"Data/PDF/{file_path}",
        extract_images_in_pdf=False,
        infer_table_structure=True,
        strategy="hi_res",
        hi_res_model_name="yolox",
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=output_path,
    )
    
    return raw_pdf_elements

# Get text summaries and table summaries
text_elements = []
table_elements = []

text_summaries = []
table_summaries = []

summary_prompt = """
Summarize the following {element_type}:
{element}
"""

def text_table_summary(raw_pdf_info):
    summary_chain = LLMChain(
        llm=ChatGroq(
            temperature = 0, 
            groq_api_key = os.environ["GROQ_API_KEY"], 
            model_name = "llama3-70b-8192"
        ),
        prompt=PromptTemplate.from_template(summary_prompt)
    )
    
    for e in raw_pdf_info:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
            summary = summary_chain.run({'element_type': 'text', 'element': e})
            text_summaries.append(summary)

        elif 'Table' in repr(e):
            table_elements.append(e.text)
            summary = summary_chain.run({'element_type': 'table', 'element': e})
            table_summaries.append(summary)

# Create Documents and Vectorstore
documents = []
retrieve_contents = []

def create_vectors(vector_name):
    for e, s in zip(text_elements, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(
            page_content=s,
            metadata={
                'id': i,
                'type': 'text',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)

    for e, s in zip(table_elements, table_summaries):
        doc = Document(
            page_content=s,
            metadata={
                'id': i,
                'type': 'table',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)
    
    vectorstore = FAISS.from_documents(documents = documents, embedding = embeddings)
    vectorstore.save_local(f"PDF/{vector_name}")
    
    return vectorstore

prompt_template = """
You are a professor and an expert in analyzing Electronics and communication subjects and topics.
Answer the question based only on the following context, which can include text, images, and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:
"""

qa_chain = LLMChain(
    llm = ChatGroq(
        temperature = 0, 
        groq_api_key = os.environ["GROQ_API_KEY"], 
        model_name = "llama3-70b-8192"
    ),
    prompt = PromptTemplate.from_template(prompt_template)
)


# Answer and question
def pdf_rag(query,file_name):
    # Check if storage already exists
    PERSIST_DIR = f"PDF/{file_name}"
    if not os.path.exists(PERSIST_DIR):
        # Getting the PDF and extracting the data (e.g., Texts, Tables, Images)
        raw_pdf_elements = get_pdf_element(f"{file_name}")
        
        # Getting the summary of Table and Text
        text_table_summary(raw_pdf_elements)
        
        # Creating Vectors
        vectorstore = create_vectors(file_name)
    else:
        # Loading the embeddings
        vectorstore = FAISS.load_local(
            f"PDF/{file_name}", 
            embeddings = embeddings,
            allow_dangerous_deserialization = True
        )
        
    relevant_docs = vectorstore.similarity_search(query)
    context = ""
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
    
    result = qa_chain.run({'context': context, 'question': query})
    
    return result
