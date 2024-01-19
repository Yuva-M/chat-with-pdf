import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tenacity

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


@tenacity.retry(stop=tenacity.stop_after_attempt(3))
def chat_with_retry(chain, inputs):
    return chain(inputs, return_only_outputs=True)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    try:
        response = chat_with_retry(chain, {"input_documents": docs, "question": user_question})
        print(response)
        st.write("Reply: ", response["output_text"])

    except GoogleGenerativeAI.generativeai.types.generation_types.StopCandidateException as e:
        # Handle the StopCandidateException here
        print(f"Generation stopped with reason: {e.finish_reason}")
        st.warning("Generation stopped. Please try again.")


def search_and_download_pdfs(folder_path, search_query):
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

    if search_query:
        pdf_files = [file for file in pdf_files if search_query.lower() in file.lower()]

    return pdf_files
def main():
    def streamlit_menu(example=2):
        if example == 2:
            # 2. horizontal menu w/o custom style
            selected = option_menu(
                key="menu_option",
                menu_title=None,
                options=["Home", "Chat with pdf", "Search"],
                icons=["house", "chat-dots", "search"],
                menu_icon="",
                default_index=0,
                orientation="horizontal",
                
            )
            return selected

    selected = streamlit_menu(example=2)
    st.markdown(
        """
        <style>
            .element-container {
                margin-top: -30px;
                padding-bottom: 30px;
            }
            
        </style>
        """,
        unsafe_allow_html=True,
    )

    container = st.container()
    

    if selected == "Home":
        
        with container:
            st.title("Welcome to Chat with PDF using Gemini 🌟")
            st.write(
                "This application allows you to chat with PDFs using Gemini, a generative AI model."
            )
            st.write(
                "1. **Chat with PDFs:** Upload your PDF files, ask questions, and get responses based on the content of the PDF."
            )
            st.write(
                "2. **Search and Download PDFs:** Search for PDFs in the directory, select one, and download it."
            )
    elif selected == "Chat with pdf":
        with container:
            st.title("Chat with PDF using Gemini 🌟")
            

            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                if not pdf_docs:
                    st.error("Please upload a PDF file first.")
                else:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                    st.success("Done")
            st.divider()

            user_question = st.text_input("Ask a Question from the PDF Files")

            if user_question:
                user_input(user_question)
                
        
                
    elif selected == "Search":
        with container:
            st.title("Search and Download PDF 🔎")
            st.caption('Can not remember the PDF name? Try searching your Professor\'s name! 😎')
          

            # Specify the folder path where PDFs are stored
            folder_path = "dir"

            # Create a search box for the user
            search_query = st.text_input("Search PDFs")
            st.divider()

            # Get the list of PDFs based on the search query
            pdf_files = search_and_download_pdfs(folder_path, search_query)

            if pdf_files:
                selected_pdf = st.selectbox("Select a PDF to download:", pdf_files)

                # Download PDF button
                if st.button("Download PDF"):
                    st.success(f"Downloading {selected_pdf}")
                    pdf_path = os.path.join(folder_path, selected_pdf)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Click here to download",
                            data=f.read(),
                            key=selected_pdf,
                            file_name=selected_pdf,
                        )
    
if __name__ == "__main__":
    main()