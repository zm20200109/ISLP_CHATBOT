from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from databases.chroma_db_connection import create_retriever
import os 
from dotenv import load_dotenv 
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-4o-mini",temperature=0)

prompt_rag = hub.pull("rlm/rag-prompt")

template = """You are a helpful assistant that generates multiple sub-questions related to the input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

def retrieve_and_rag(question,prompt_rag,sub_question_generator_chain,retriever):
    sub_questions = sub_question_generator_chain.invoke({"question":question})
    rag_results = []
    
    for sub_question in sub_questions:
        retrieved_docs = retriever.get_relevant_documents(sub_question)
      
        answer = (prompt_rag | chat | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
    return rag_results,sub_questions


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | chat 
    | StrOutputParser()
)

def rag_with_sq_with_individual_ans(question, retriever):
      generate_queries_decomposition = ( prompt_decomposition | chat | StrOutputParser() | (lambda x: x.split("\n")))
      answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition,retriever)
      context = format_qa_pairs(questions, answers) 
      final_answer = final_rag_chain.invoke({"context":context,"question":question})
      return final_answer 

if __name__ == "__main__":
    pass
    # question = "Tell me all classification metrics."
    # chunks_path = "results/chunked_elements_1.pkl" 
    # txt_summaries_path = "results/1_txt_summaries.pkl"
    # tbl_summaries_path = "results/1_table_summaries.pkl"
    # retriever = create_retriever(chunks_path,txt_summaries_path,tbl_summaries_path,"similarity",5)
    # generate_queries_decomposition = ( prompt_decomposition | chat | StrOutputParser() | (lambda x: x.split("\n")))
    # answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition,retriever)
    # context = format_qa_pairs(questions, answers) 
    # final_answer = final_rag_chain.invoke({"context":context,"question":question})
    # print(final_answer)
