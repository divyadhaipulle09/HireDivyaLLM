

import json
import ast
import time
import pandas as pd
import streamlit as st
import openai
from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangPinecone
from langchain.chat_models import ChatOpenAI
from streamlit_utils import sidebar
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Constants and setup
MODEL = 'text-embedding-ada-002'
INDEX_NAME = 'pinecode-index'
NAMESPACE = 'HireDivya'
SPEC = ServerlessSpec(cloud="aws", region="us-east-1")  # Ensure the region is supported by the free plan
SYSTEM_PROMPT = (
    """### Who are You? ###
    You are a resume assessment expert.
    ### What is your job? ###
    You answer queries from data science recruiters regarding a candidate profile using the context given below.
    ### Context: ### {context}
    ### INSTRUCTIONS: ###
    Do not blindly use the context as the answer. Please frame the answer in a user-readable format and make sure it perfectly answers the input.
    Feel free to rephrase the answer based on the input and the context as evidence.
    If you do not know any question, and if there is no context provided, simply say that you do not have enough information.
    """
)

# Initialize Pinecone client and Streamlit
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
st.title("Get to know Divya Dhaipullay :)")

# Sidebar and session state
def initialize_sidebar():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    sidebar()

initialize_sidebar()

def ensure_index():
    # Check if the index exists
    try:
        index = pc.Index(INDEX_NAME)
        index.describe_index_stats()
        return index
    except Exception:
        # Create the index if it doesn't exist
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=SPEC
            )
            return pc.Index(INDEX_NAME)
        except Exception as e:
            st.error(f"Failed to create or access Pinecone index: {e}")
            return None

# Optimize embedding creation with caching
@st.cache_resource
def create_embeddings(texts):
    llm_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    embeddings_list = [llm_client.embeddings.create(input=[text], model=MODEL).data[0].embedding for text in texts]
    return embeddings_list

def process_pdf(filepath):
    pdf = PyPDFLoader(filepath)
    data = pdf.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    return [str(doc).replace("\\n●", " ").replace("\\n", " ").replace("•", " ") for doc in docs]

# Load embeddings from file or create new ones
def load_or_create_embeddings(index, pdf_path):
    try:
        # Load existing embeddings if available
        embedding_df = pd.read_csv('embeddings.csv')
        embeddings = embedding_df["vectors"].apply(lambda x: list(map(float, ast.literal_eval(x)))).to_list()
        text = embedding_df["text"].tolist()
    except FileNotFoundError:
        # If embeddings file not found, process the PDF and create new embeddings
        text = process_pdf(pdf_path)
        embeddings = create_embeddings(text)
        embedding_df = pd.DataFrame({
            "vectors": embeddings,
            "text": text
        })
        embedding_df.to_csv('embeddings.csv', index=False)
    
    # Upsert the vectors into the Pinecone index
    vectors = [{"id": str(i), "values": embeddings[i], "metadata": {"text": text[i]}} for i in range(len(text))]
    index.upsert(vectors=vectors, namespace=NAMESPACE)


def query_LLM(index, user_query):
    openai_embed = OpenAIEmbeddings(model=MODEL, api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = LangPinecone(index, openai_embed.embed_query, "text", namespace=NAMESPACE)
    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model_name='gpt-3.5-turbo', temperature=0.0)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    response = rag_chain.invoke({"input": user_query})
    
    return response["answer"]

def query_LLM_with_retry(index, user_query, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return query_LLM(index, user_query)
        except openai.error.RateLimitError as e:
            if attempt < retries - 1:
                print(f"Rate limit exceeded, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                st.error(f"Failed after {retries} attempts: {e}")
                return "I'm currently unavailable due to API rate limits. Please try again later."
def main():
    index = ensure_index()
    if index:
        load_or_create_embeddings(index, "HireDivyaResume.pdf")
        
        # Display previous conversation history
        if st.session_state.messages:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input with a unique key
        if prompt := st.chat_input("What is up?", key=f"chat_input_{len(st.session_state.messages)}"):
            # Append user message to session state
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response from LLM
            response = query_LLM(index, prompt)
            
            # Append assistant response to session state
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

main()
