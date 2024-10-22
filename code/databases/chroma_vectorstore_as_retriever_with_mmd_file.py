from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain_chroma import Chroma 
import os 
from dotenv import load_dotenv

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
def create_splits(split_chunks, chunk_size=1500,chunk_overlap=30):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = chunk_size, chunk_overlap = chunk_overlap 
  )
  return text_splitter.split_documents(split_chunks)


def create_retriever(splits):
  vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
  retriever = vectorstore.as_retriever()
  return retriever


def split_mmd_by_headers(fpath, chunk_size=1500, chunk_overlap=30):
  if fpath == None:
    print(os.getcwd())
    mmd_file_path = "./data_dir/ISLP.mmd"
  else:
    mmd_file_path = fpath
  with open(mmd_file_path, "r", encoding="utf-8") as file:
      ISLP_mmd = file.read()
  headers_to_split_on = [
    ('##', 'Header1'),
    ('###', 'Header2'),
    ('####','Header3'), # Python komande su date sa * 
  ]
  mmd_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
  split_chunks = mmd_header_splitter.split_text(ISLP_mmd)
  splits = create_splits(split_chunks, chunk_size, chunk_overlap)
  retriever = create_retriever(splits)
  return retriever 

