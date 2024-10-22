from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from databases.chroma_db_connection import create_retriever
from pydantic import BaseModel 
from typing import Any 
from pydantic import Field 
from  langchain.load import dumps,loads 
from operator import itemgetter
from langchain_core.runnables import RunnableLambda 
import os 
from dotenv import load_dotenv 
load_dotenv() 

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=OPEN_API_KEY,model="gpt-4o-mini",temperature=0)


class Element(BaseModel):
    type:str = Field(description="Type of pdf documents' element.")
    text:Any = Field(description="This is text content of pdf document's element")    
    page_no:int = Field(description="Page number of the original document from where chunk belongs.")


# data model for structured output 

class GradeDocument(BaseModel):
    """Checks if document is relevant for asked question."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = chat.with_structured_output(GradeDocument)
system_grader = """ 
You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader),
        ("human","Retrieved document: \n\n {document} \n\n User question: {question}"),  
    ],
)
retrieval_grader = grade_prompt | structured_llm_grader

def grade_retrieved_documents(input_dictionary:dict):
    question = input_dictionary['question']
    documents = input_dictionary['context_documents']
    filtered_documents = [] 
    for d in documents: 
        score = retrieval_grader.invoke({"question":question,"document":d})
        if score.binary_score == "yes":
            filtered_documents.append(d)
        else:
            continue
    return {"context_documents":filtered_documents,"question":question}


def get_unique_union(documents: list[list]): 
    """Method creates set of retrieved documents."""
    all_docs = [dumps(doc) for sublist in documents for doc in sublist] 
    unique_docs = list(set(all_docs))
    return [loads(doc) for doc in unique_docs]

class MultiQuery(BaseModel):
    """Make a list of different perspectives of initial query."""
    queries: list[str] = Field(
        description="Given a user question make five different perspectives of initial query."
    )

def reciprocal_rank_fusion(results: list[list], k=60):
    """This method ranks documents based on the formula for fused scores and their individual rank from each question and optional parameter k"""
    fused_scores = {}
    for documents in results:
        for rank, document in enumerate(documents):
            document_string = dumps(document)
            if document_string not in fused_scores: 
                fused_scores[document_string] = 0 
            previous_score = fused_scores[document_string] 
            fused_scores[document_string] += 1/(rank + k) 
    
    reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda el: el[1], reverse=True)]
    return reranked_results

system = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question} 
"""

fusion_template = ChatPromptTemplate.from_messages([
    ("system",system),
    ("human","{question}")
])

rag_fusion_llm = chat.with_structured_output(MultiQuery)
query_generation_chain = (
    fusion_template 
    | rag_fusion_llm
)

def multiquery_retrieval_chain(retriever):
    return (query_generation_chain 
                |(lambda x: x.queries)
                | retriever.map() 
                | reciprocal_rank_fusion 
                ) 

final_chain_template = """
You don't have any knowledge about statistics and machine learning. 
You are only allowed to answer the questions based on the data from provided context.
Also, you are not allowed to fix answers based on your knowledge which is not provided from the context.
Your answer must be completely created from the retrieved context.
If answer is not contained in your context, you will tell to the user that you don't know the answer and sugest to the user to be more specific.
\n
Allowed context for answering the user question:
\n
\n
{context_documents}
\n
\n
Question: {question}
"""
final_prompt = ChatPromptTemplate.from_template(final_chain_template)

def final_chain(retriever):
    return (
        {"context_documents":multiquery_retrieval_chain(retriever),
        "question":itemgetter("question")}
        | RunnableLambda(grade_retrieved_documents) 
        | final_prompt
        | chat 
        | StrOutputParser() 
    )
    
if __name__=="__main__":
    chunks_path = "results/chunked_elements_1.pkl" 
    txt_summaries_path = "results/1_txt_summaries.pkl"
    tbl_summaries_path = "results/1_table_summaries.pkl"
    retriever = create_retriever(chunks_path,txt_summaries_path,tbl_summaries_path,"similarity",5)
    #question = "Tell me all classification metrics."
    #question = "Retrieve formula for standardizing predictors."
    question = "Tell me formula for multivariate gaussian distribution." 
    answer = final_chain(retriever).invoke({"question":question})
    print(answer)
