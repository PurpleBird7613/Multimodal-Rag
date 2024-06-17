# Multimodal RAG Application🖥️📚🤖

Welcome to the Multimodal RAG (Retrieve-and-Generate) Application! This project enables you to chat with your documents, asking questions, requesting summaries, and receiving detailed explanations based on the contents of PDF, CSV, and Excel files.

## Features✨

- *Multimodal Document Support:* Interact with PDF, CSV, and Excel files.
- *Natural Language Processing:* Ask anything about your documents and get relevant responses.
- *Summarization:* Request brief summaries or detailed explanations.
- *Diagrams:* Can generate graphs,charts for the CSV,Excel data.
- *Interactive Q&A:* Pose specific questions regarding the content of your documents.

## Installation📜

To get started with the Multimodal RAG Application, follow these steps:

1. *Clone the Repository:*

    ```
    git clone https://github.com/PurpleBird7613/Multimodal-Rag.git
    cd Multimodal-Rag
    ```

2. *Run the shell script:*

    ```
    chmod +x installer.sh
    ./installer.sh
    ```  

4. *Set Up API Key:*

    The application requires an API key to function. You need to set the GROQ_API_KEY inside the .env file.
    
    ```
    GROQ_API_KEY=your_api_key_here
    ```
    

5. *Run the Application:*

    ```
    streamlit run main.py
    ```
    

## Usage📖

1. *Upload Documents:*
   - Open the Streamlit app in your browser.
   - Use the upload button to upload PDF, CSV, or Excel files.

2. *Interact with Your Documents:*
   - Once the documents are uploaded, use the chat interface to ask questions.
   - Request summaries or explanations by typing your query.
   - Receive instant responses based on the content of the uploaded documents.
