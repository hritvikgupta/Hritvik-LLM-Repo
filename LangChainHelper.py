from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import requests
import tempfile
import os
from InstructorEmbedding import INSTRUCTOR
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import faiss  # Ensure FAISS is correctly imported
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from langchain.chains import RetrievalQA
import google.generativeai as genai

from pydantic import BaseModel, Field
import google.generativeai as genai
load_dotenv()
import os

file_path = "/Users/hritvikgupta/Downloads/Hritvik-LLM-main/HQA.csv"
api_key = os.environ["GOOGLE_API_KEY"]

instructor_embeddings = HuggingFaceInstructEmbeddings()
vector_db_file_path = "/Users/hritvikgupta/Downloads/Hritvik-LLM-Main/my_index.faiss"
def create_vector_db():
    loader = CSVLoader(file_path=file_path, source_column = "Prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding = instructor_embeddings)
    vectordb.save_local(vector_db_file_path)
def download_faiss_index(github_raw_url, local_file_path):
    response = requests.get(github_raw_url)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception("Failed to download FAISS index from GitHub")

# Define the folder name where you want to store the downloaded files
folder_name = "downloaded_files"

# Create the full path to the folder in the current working directory
download_folder_path = os.path.join(os.getcwd(), folder_name)

# Create the folder if it doesn't exist
os.makedirs(download_folder_path, exist_ok=True)

# URL of the raw FAISS index file in your GitHub repository
github_raw_faiss = "https://github.com/hritvikgupta/Hritvik-LLM-Repo/raw/main/my_index.faiss/index.faiss"
github_raw_pkl = "https://github.com/hritvikgupta/Hritvik-LLM-Repo/raw/main/my_index.faiss/index.pkl"

# Create paths for the downloaded files inside the folder
faiss_index_file_path = os.path.join(download_folder_path, "index.faiss")
pkl_index_file_path = os.path.join(download_folder_path, "index.pkl")

# Download and save the FAISS index files
download_faiss_index(github_raw_faiss, faiss_index_file_path)
download_faiss_index(github_raw_pkl, pkl_index_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(download_folder_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    custom_llm = CustomLLM('gemini-pro', api_key)
    chain = RetrievalQA.from_chain_type(
        llm=custom_llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True
    )

    return chain

class CustomLLM(LLM, BaseModel):
    llm: Any = Field(default=None)  # Declare 'llm' as a field with default value None

    def __init__(self, model_name, api_key):
        super().__init__()
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(model_name)

    def _llm_type(self) -> str:
        return "genai-gemini-pro"

    def _call(
    self,
    prompt: str,
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        response = self.llm.generate_content(prompt)  # Call the generative model

        # Correctly accessing the response text
        if hasattr(response, '_result') and hasattr(response._result, 'candidates'):
            if response._result.candidates and len(response._result.candidates) > 0:
                candidate = response._result.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    response_text = candidate.content.parts[0].text
                else:
                    raise TypeError("Unexpected structure in response candidate content")
            else:
                raise TypeError("No candidates in response")
        else:
            raise TypeError("Unexpected response structure")

        return response_text


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    
if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    print(chain("Does he require any visa sponsership?"))