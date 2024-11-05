from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore 
from langchain_chroma import Chroma 
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
from pydantic import Field, BaseModel
from typing import Any 
import os 
from langchain_core.documents.base import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
import random 
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma 
from langchain_core.documents.base import Document
from dotenv import load_dotenv
load_dotenv()


class Element(BaseModel):
    type:str = Field(description="Type of pdf documents' element.")
    text:Any = Field(description="This is text content of pdf document's element")    
    page_no:int = Field(description="Page number of the original document from where chunk belongs.")

def load_pickle_file(fpath):
  with open(fpath,"rb") as file:
    return pickle.load(file)


def create_parent_retriever(txt_chunks_path, tbl_elements_path,k_parameter): # txt i tabele 
  """
  txt_chunks_path = "./data_dir/chunked_elements_1.pkl", tbl_elements_path = "./data_dir/raw_table_elements.pkl"
  """
  txt_chunks = load_pickle_file(txt_chunks_path)
  text_elements = [Element(type="text",text=element.text,page_no=dict(element.metadata.fields)['page_number']) for element in txt_chunks]
  tbl_elements = load_pickle_file(tbl_elements_path)

  # documents
  txt_docs = [Document(page_content=element.text, metadata={"page_no":element.page_no}) for element in text_elements]
  tbl_docs = [Document(page_content=element.text, metadata={"page_no":element.page_no}) for element in tbl_elements]

  all_docs = txt_docs + tbl_docs 

  child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
  vectorstore = Chroma(
     collection_name="full_documents_3", embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
  ) # za children, ovde su sacuvani subdocs!!!
  store = InMemoryStore() # za parent document
  retriever = ParentDocumentRetriever(
      vectorstore=vectorstore,
      docstore=store,
      child_splitter=child_splitter,
  )
  retriever.add_documents(all_docs, ids=None)
  retriever.search_kwargs = {"k":k_parameter,}
  return retriever 


def create_retriever_with_cc(txt_chunks_path, tbl_elements_path,k_parameter): # txt i tabele 
  """
  txt_chunks_path = "./data_dir/chunked_elements_1.pkl", tbl_elements_path = "./data_dir/raw_table_elements.pkl"
  """
  txt_chunks = load_pickle_file(txt_chunks_path)
  text_elements = [Element(type="text",text=element.text,page_no=dict(element.metadata.fields)['page_number']) for element in txt_chunks]
  tbl_elements = load_pickle_file(tbl_elements_path)

  # documents
  txt_docs = [Document(page_content=element.text, metadata={"page_no":element.page_no}) for element in text_elements]
  tbl_docs = [Document(page_content=element.text, metadata={"page_no":element.page_no}) for element in tbl_elements]

  all_docs = txt_docs + tbl_docs 

  retriever = Chroma.from_documents(
    all_docs,OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
  ).as_retriever()
  

  # sada ce iz dokumenata koje je vratio base retriever compresor da izvuce relevantne info
  llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
  compressor = LLMChainExtractor.from_llm(llm)
  compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor, base_retriever=retriever
  )
  return compression_retriever  

def create_ensemble_retriever(txt_chunks_path, tbl_elements_path,k_parameter): # txt i tabele 
  """
  txt_chunks_path = "./data_dir/chunked_elements_1.pkl", tbl_elements_path = "./data_dir/raw_table_elements.pkl"
  """
  txt_chunks = load_pickle_file(txt_chunks_path)
  text_elements = [Element(type="text",text=element.text,page_no=dict(element.metadata.fields)['page_number']) for element in txt_chunks]
  tbl_elements = load_pickle_file(tbl_elements_path)

  # documents
  txt_docs = [Document(page_content=element.text, metadata={"page_no":element.page_no}) for element in text_elements]
  tbl_docs = [Document(page_content=element.text, metadata={"page_no":element.page_no}) for element in tbl_elements]

  all_docs = txt_docs + tbl_docs 
  random.shuffle(all_docs)
  half_size = len(all_docs) // 2
  list1 = all_docs[:half_size]
  list2 = all_docs[half_size:]
  
  list1 = [elem.page_content for elem in list1]
  #list2 = [elem.page_content for elem in list2]
  # sparse retriever 
  bm25_retriever = BM25Retriever.from_texts(
    list1
  )
  bm25_retriever.k = k_parameter 

  # dense retriever 
  chroma_retriever = Chroma.from_documents(
    list2, OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
  ).as_retriever()
  chroma_retriever.search_kwargs = {"k":k_parameter,}
  
  ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
  )
  return ensemble_retriever 






