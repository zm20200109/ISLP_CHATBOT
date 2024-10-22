from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from databases.chroma_db_connection import create_retriever
import os 
from dotenv import load_dotenv
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-4o-mini",temperature=0)


template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)



def return_retrieval_chain(retriever):
  return (prompt_hyde | chat | StrOutputParser() ) | retriever 

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | chat
    | StrOutputParser()
)

def HyDE_chain(question,retriever):
  retrieval_chain = (prompt_hyde | chat | StrOutputParser() ) | retriever 
  answer = final_rag_chain.invoke({"context":retrieval_chain,"question":question})
  return answer 

if __name__ == "__main__":
    question = "Tell me all classification metrics."
    answer = final_rag_chain.invoke({"context":retrieval_chain,"question":question})
    print(answer)