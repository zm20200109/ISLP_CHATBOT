from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from databases.chroma_db_connection import create_retriever
from langchain_core.runnables import RunnableLambda 
import os 
from dotenv import load_dotenv 
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-4o-mini",temperature=0) #o-mini



examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question,
             which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)


# Response prompt 
response_prompt_template = """
You are an expert of world knowledge. I am going to ask you a question. 
Your response should be comprehensive and not contradicted with the following context if they are relevant. 
Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

def step_back_chain(question,retriever):
    chain = (
        {
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            "step_back_context": prompt | chat | StrOutputParser() | retriever,
            "question": lambda x: x["question"],
        }
        | response_prompt
        | chat 
        | StrOutputParser()
    )
    return chain.invoke({"question": question})

if __name__ == "__main__":
    chunks_path = "results/chunked_elements_1.pkl" 
    txt_summaries_path = "results/1_txt_summaries.pkl"
    tbl_summaries_path = "results/1_table_summaries.pkl"
    retriever = create_retriever(chunks_path,txt_summaries_path,tbl_summaries_path,"similarity",5)
    question = "Tell me all classification metrics."
    answer = step_back_chain(question,retriever)
    print(answer)