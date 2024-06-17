import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import sys
import os


# Ensure data directory exists
os.makedirs("Data/CSV", exist_ok=True)

# Setting LLM
llm = ChatGroq(
    temperature = 0,
    model_name = "llama3-70b-8192", 
    api_key = os.environ["GROQ_API_KEY"]
)

# Setting emmbedings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#+++++++++ Pandas AI - For ploting graph,charts ++++++++++++
def plot_diagram(file_name,prompt):
    df = pd.read_csv(f"Data/CSV/{file_name}")
    df = SmartDataframe(df, config={"llm": llm})

    df.chat(prompt)
    
    
#+++++++++++++++ Vector Store method ++++++++++++++++++++++
# loading and spliting into chuncks of csv data
def get_csv_data(file_name):
    loader = CSVLoader(file_path = f"Data/CSV/{file_name}", encoding = "utf-8", csv_args = {'delimiter': ','})
    data = loader.load()
    #print(data)

    # Split the text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(data)
    #print(len(text_chunks))
    
    return text_chunks

# Storing as vector
def csv_vector_store(text_chunks,DB_FAISS_PATH): 
    # Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    vectorstore.save_local(f"CSV/{DB_FAISS_PATH}")
    
    return vectorstore

# CSV Rag
def csv_rag(query,file_name):
    # Check if storage already exists
    PERSIST_DIR = f"CSV/{file_name}"
    if not os.path.exists(PERSIST_DIR):
        # getting the csv file data
        text_chunks = get_csv_data(file_name)

        # Creating vector
        vectorstore = csv_vector_store(text_chunks,file_name)
    else:
        # Loading the vectors
        vectorstore = FAISS.load_local(
            f"CSV/{file_name}", 
            embeddings = embeddings,
            allow_dangerous_deserialization = True
        )
    
    chat_history = []
    
    qa = ConversationalRetrievalChain.from_llm(llm, retriever = vectorstore.as_retriever())
    response = qa({"question":query, "chat_history":chat_history})
    
    plot_diagram(file_name,f"{query} - {response['answer']}")
    
    return response['answer']
