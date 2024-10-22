import pymongo 
import requests 
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
import pickle 
import uuid 
from pydantic import BaseModel 
from typing import Any 
from pydantic import Field 
import os 
from dotenv import load_dotenv 

load_dotenv() 

#zmahovac je username, a pass je gUwtDWG5bgJrGxep
# mongodb+srv://zmahovac:<db_password>@cluster0.ictm8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0 konekcioni string 

client = pymongo.MongoClient("mongodb+srv://zmahovac:gUwtDWG5bgJrGxep@cluster0.ictm8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",connect=False)
db = client.rag_db 
collection = db.chunk_embeddings

class Element(BaseModel):
    type:str = Field(description="Type of pdf documents' element.")
    text:Any = Field(description="This is text content of pdf document's element")    
    page_no:int = Field(description="Page number of the original document from where chunk belongs.")

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

    
def semantic_search(query:str):
    results = collection.aggregate([
    {"$vectorSearch":{
        "queryVector":generate_embeddings(query),
        "path":"summary_embedding",
        "numCandidates":1531,
        "limit":5,
        "index":"ChunkSemanticSearch"
        }
    }
    ]);
    documents = []
    for document in results:
        documents.append(document['original'])
    return documents

    

def generate_embeddings(text:str)->list[float]:
    embedding = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)
    return embedding.embed_query(text)

def get_data(fpath):
    with open(fpath, 'rb') as file: 
        data = pickle.load(file)
    return data  

def insert_txt_data():

    txt_summaries = get_data('results/1_txt_summaries.pkl')
    txt_originals = get_data('results/raw_txt_elements.pkl')
    text_ids = [str(uuid.uuid4()) for _ in txt_originals] 
    for idx,summary in enumerate(txt_summaries):
        document = {
            "id":text_ids[idx],
            "original":txt_originals[idx].text,
            "summary":summary,
            "summary_embedding":generate_embeddings(summary)
        }
        collection.insert_one(document)

def insert_table_data():
    tbl_summaries = get_data('results/1_table_summaries.pkl')
    tbl_originals = get_data('results/raw_table_elements.pkl')
    tbl_ids = [str(uuid.uuid4()) for _ in tbl_summaries]
    for idx, tbl_summary in enumerate(tbl_summaries):
        document = {
            "id":tbl_ids[idx],
            "original":tbl_originals[idx].text,
            "summary":tbl_summary,
            "summary_embedding":generate_embeddings(tbl_summary)
        }
        collection.insert_one(document)

def insert_img_data():
    img_summaries = get_data("results/1_image_summaries.pkl")
    counter = 0 
    for element in img_summaries:
        print(counter)
        document = {
            "id":element[0],
            "original":element[1],
            "summary":element[1],
            "summary_embedding":generate_embeddings(element[1])
        }
        collection.insert_one(document)
        counter += 1

def insert_formulas_data():
    formula_descriptions = get_data('results/formula_descriptions.pkl')
    counter = 0 
    for element in formula_descriptions:
        print(counter)
        document = {
            "id":element[0],
            "original":element[1],
            "summary":element[1],
            "summary_embedding":generate_embeddings(element[1])
        }
        collection.insert_one(document)
        counter += 1


if __name__ == '__main__':
    #query = "What is a studentized residual, and how is it used to identify outliers?"
    #documents = semantic_search(query)
    #print(documents)
    insert_formulas_data()