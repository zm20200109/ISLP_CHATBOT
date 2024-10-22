from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from databases.chroma_db_connection import create_retriever
import os 
from dotenv import load_dotenv 
load_dotenv() 

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=OPEN_API_KEY,model="gpt-4o-mini",temperature=0)

##############################    RECURSIVE DECOMPOSITION CHAIN     #######################################################################################################################################################################################

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = ( prompt_decomposition | chat | StrOutputParser() | (lambda x: x.split("\n")))

def decompose_query(question):
    questions = generate_queries_decomposition.invoke({"question":question})
    return questions 

# Prompt
template = """
You have no knowledge of statistics, mathematical concepts, or machine learning, except for the content provided to you.

Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here are any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use only the above context and any background question + answer pairs to answer the question: \n {question}
If there is no suitable answer in the provided context, simply say that you don't have that information.
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def query_decomposition_recursive_answering_chain(question, retriever):
    questions = decompose_query(question)
    q_a_pairs = ""
    for q in questions:
    
        rag_chain = (
            {"context": itemgetter("question") | retriever, 
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs")} 
            | decomposition_prompt
            | chat 
            | StrOutputParser())

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    return answer #q_a_pairs 


#########################################       MAIN        #####################################################################################################################################

if __name__ =='__main__':
     
    chunks_path = "results/chunked_elements_1.pkl" 
    txt_summaries_path = "results/1_txt_summaries.pkl"
    tbl_summaries_path = "results/1_table_summaries.pkl"
    retriever = create_retriever(chunks_path,txt_summaries_path,tbl_summaries_path,"similarity",5)
    #question = "Tell me all classification metrics."
    question = "Retrieve formula for standardizing predictors."
    # radi 
    answer = query_decomposition_recursive_answering_chain(question,retriever) 
    print(answer)


    