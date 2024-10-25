import sys
from google.colab import drive
from google.colab import auth
from google.cloud import storage
from images_retrieval.images_retrieval import load_index, load_model_and_index
from formulas_retrieval.formulas_retrieval import load_formulas_index, load_formulas_model_and_index
from databases.chroma_db_connection import create_retriever
from text_and_tables_retrieval.multiquery_startegies.subquestions_with_individual_answers_chain import rag_with_sq_with_individual_ans
from text_and_tables_retrieval.multiquery_startegies.multiquery_chain import final_chain as final_chain_mq
from text_and_tables_retrieval.multiquery_startegies.HyDE_chain import HyDE_chain
from text_and_tables_retrieval.multiquery_startegies.step_back_chain import step_back_chain
from text_and_tables_retrieval.multiquery_startegies.decomposition_with_recursive_answering_chain import query_decomposition_recursive_answering_chain
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import getpass
from typing import Annotated
from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint 
from io import BytesIO
from google.oauth2 import service_account
from google.cloud import storage
from databases.chroma_vectorstore_as_retriever_with_mmd_file import split_mmd_by_headers
import gradio as gr 
from dotenv import load_dotenv


load_dotenv() 

def get_image(results):
    blob_name = results.split("figures/")[1]
    credentials_path = os.getenv("GOOGLE_CLOUD_KEY_FILE")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket("images_data_bucket")
    blob = bucket.blob(blob_name)
    image_data = blob.download_as_bytes()
    return io.BytesIO(image_data)

# 1. Ucitavanje indeksa
#auth.authenticate_user() # ovo radim pozivom svake metode za get image 

# ucitavanje indexa za slike
images_index = load_model_and_index()
# ucitavanj indexa za formule
formulas_index = load_formulas_model_and_index()
# ucitavanje multi vector retrievera za text


chunks_path = "./data_dir/chunked_elements_1.pkl" #"/content/drive/MyDrive/project_work/code/results/chunked_elements_1.pkl"#"results/chunked_elements_1.pkl" # chunk_by_title strategy, combine_text_under_n_chars=1000,max_characters=1600, multipage_sections=True, new_after_n_chars=1200, overlap=True,
txt_summaries_path = "./data_dir/1_txt_summaries.pkl"#"/content/drive/MyDrive/project_work/code/results/1_txt_summaries.pkl"#"results/1_txt_summaries.pkl"
tbl_summaries_path = "data_dir/1_table_summaries.pkl" #"results/1_table_summaries.pkl"

retriever = create_retriever(chunks_path,txt_summaries_path,tbl_summaries_path,"similarity",5)

retriever_from_vectorstore_for_mmd = split_mmd_by_headers(None, chunk_size=1500, chunk_overlap=30)

# za tekstualne odgovore korisnik moze da odabere rag sa individualnim odgovorima za svako dekomponovano pitanje. 
def rag_with_sq_with_individual_ans_chain(question, retriever):
  return rag_with_sq_with_individual_ans(question,retriever)

# za txt odgovore korisnik moze da odabere rag sa multiquery chain 
def answer_with_multiquery_chain(question, retriever):
  return final_chain_mq(retriever).invoke({"question":question})

# HyDE chain: 
def answer_with_HyDE_chain(question,retriever):
  return HyDE_chain(question,retriever)

# step back 
def answer_with_stepback_chain(question, retriever_from_vectorstore_for_mmd): # retriever 
  return step_back_chain(question,retriever)

# rekurzivno odgovaranje 

def answer_decomposition_recursive(question, retriever):
  return query_decomposition_recursive_answering_chain(question,retriever)

# mapiranje odgovora 

answering_methods = {
    "Query Decomposition (Individual Answers)": rag_with_sq_with_individual_ans_chain,
    "Multiquery Chain": answer_with_multiquery_chain,
    "HyDE Chain": answer_with_HyDE_chain,
    "Step Back Chain": answer_with_stepback_chain,
    "Recursive Decomposition": answer_decomposition_recursive,
}

selected_retriever = None 

def select_retriever(selected_option):
  global selected_retriever 
  if selected_option == "RAG with ISLP.mmd + vectostore as retreiver":
    selected_retriever = retriever_from_vectorstore_for_mmd
  else:
    selected_retriever = retriever 


# Funkcija koja poziva odgovarajuću metodu
def get_answer(question, method):
    answer_function = answering_methods.get(method)
    if answer_function:
        return format_text(answer_function(question,selected_retriever)) # ovo je dostupni ucitanji retriever 
    return "Invalid method selected."


# 2. Kreiranje grafa 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini",temperature=0)

class IntentClassifier(BaseModel):
  """The title of the next node which is going to be executed in graph based on the user's question."""
  intent: str = Field(
      description="The following operation is connected to the retrieval of the formula, text or image, available values are 'formula', 'text', 'image'"
  )

structured_llm_intent_classifier = llm.with_structured_output(IntentClassifier)

system = """You are an assistant who has the task to classify user's question into following classes:
image - if user asks you to be provided with figure, graph, image or graphical representation \
formula - if user asks you to be provided with formula of specific statistic/mathematic/machine learning concept \
text - if there is no need for user to be provided with formula or image.
Possible range of answers: formula, text, image.
"""

classify_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human","User question: {question}") # nije dostupno pitanje jos
])

# ovaj retriever pozivas u node-u
classify_intent_retriever = classify_prompt | structured_llm_intent_classifier


formula_text = """Here is the question you need to answer:

\n --- \n {question} \n --- \n
Here is additional context relevant to the question which is connected to the retrieved formula:

\n --- \n {context} \n --- \n
Here is retrieved formula:
\n --- \n {formula} \n --- \n

Use the context above to find the context of the formula. You are only allowed to answer the question based on the provided context. You don't have any further knowledge about
machine learning, statistical and matematical concepts provided in the context.
"""

formula_text_prompt = ChatPromptTemplate.from_template(formula_text)



image_text_2 = """
Your task is to generate the best description of figure from the book based on the following question:
\n --- \n{question} \n --- \n
You are provided with the retrieved context: 
\n --- \n{context} \n --- \n
"""

image_text_prompt_2 = ChatPromptTemplate.from_template(image_text_2) 

result_example = r"\text{{MSE}} = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} (y_i - \hat{{y}}_i)^2"

formula_to_latex_template = """Your task is to transform following formula to the LaTeX format.
Here is your formula: {formula}.""" + f"""Your answer must only consists of  markdown representation of the formula
without any other supplementary content. You are not allowed to use your knowledge except formula provided to you.
This is example of the format of the result: {result_example}"""

formula_to_latex_prompt = ChatPromptTemplate.from_template(formula_to_latex_template)


text_to_latex_template = r"""Your task is to transform text to the LaTex format.
You must label inline LaTeX with the following LaTeX delimeters: left:\(, right: \).
If mathematic formulas are provided, they must be shown in the separate line using
following LaTeX delimeters: left:\[, right:\]. Return only final LaTeX data without supplementary content. 
You are not allowed to use your knowledge except textual document provided to you in the text.
Here is text: {text}
"""

text_to_latex_prompt = ChatPromptTemplate.from_template(text_to_latex_template)

class State(TypedDict):
  messages:Annotated[list, add_messages]
  question: str
  formulas: List[dict] # lista formula koje su vracene iz indexa za formule
  images: List[dict] # lista slika koje su vracene iz indexa za slike
  texts: List[str] # lista textova koji su vraceni iz chroma vectorstore - a
  next_operation: str # formula, image, text, history, end? - ali hajde prvo ova 3
  intent: str  # formula, image, text, other - na primer
  answer: str
  image: io.BytesIO


def classify_intent(state:State): #state: State
  question = state["question"]
  answer = classify_intent_retriever.invoke({"question":question})
  return {"intent":answer.intent}

def formulas_retriever(state: State):
  question = state["question"]
  formulas = formulas_index.search(question)
  return {"question":question,"formulas":[formulas[0]]}

def images_retriever(state: State):
  question = state["question"]
  images =  images_index.search(question)
  return {"question":question, "images":[images[0]]}

def text_retriever(state: State):
  global CURRENT_TEXT 
  question = state["question"]
  documents = retriever.invoke(question)
  answer = final_chain_mq(retriever).invoke({"question":question})
  CURRENT_TEXT = answer 
  return {"question":question, "texts":[documents[0]], "answer":answer}

def decide_next_intent(state:State):
  if state["intent"] == "formula":
    return "formulas_index"
  elif state["intent"] == "image":
    return "images_index"
  else:
    return "chroma_index"

CURRENT_IMAGE = None 
CURRENT_FORMULA = None 
CURRENT_TEXT = None 

def show_image(state:State):
  global CURRENT_IMAGE 
  image_stream = state["image"]
  answer = state["answer"]
  image = Image.open(image_stream)
  CURRENT_IMAGE = image_stream
  print(f"STREAM:{CURRENT_IMAGE}") 
  plt.imshow(image)
  plt.axis("off")
  plt.show()
  return {"answer":answer}


def transform_formula_to_latex(formula):
  chain = (
    formula_to_latex_prompt 
    | llm 
    | StrOutputParser()
  )
  answer = chain.invoke({"formula":formula})
  return answer

def transform_text_to_latex(text):
  chain = (
    text_to_latex_prompt
    | llm
    | StrOutputParser()
  )
  answer = chain.invoke({"text":text})
  return answer 


def search_for_formula_context(state:State):
  global CURRENT_FORMULA
  global CURRENT_TEXT
  context = state["formulas"][0]["content"]  # za retriever
  question = state["question"] # originalno pitanje
  formula = state["formulas"][0]["content"] # formula
  CURRENT_FORMULA = transform_formula_to_latex(formula)
  print("FORMULA:",formula)
  context_data = retriever.invoke(context)

  chain = (
           formula_text_prompt
          | llm
          | StrOutputParser())
  answer = chain.invoke({
            "context":context_data[0], # izlaz iz retrievera
            "question":question,
            "formula":formula
            })
  CURRENT_TEXT = transform_text_to_latex(answer)
  return {"answer":answer, "formulas":[formula]}


def search_for_image_context(state:State):
  global CURRENT_TEXT 
  question = state["question"]
  desc = state["images"][0]["content"]
  metadata = state["images"][0]["document_metadata"]["metadata"] 

  image_context = retriever_from_vectorstore_for_mmd.invoke(question) # desc
  chain = (
      image_text_prompt_2 #image_text_prompt
      | llm
      | StrOutputParser()
  )
  answer = chain.invoke({
      "context":image_context[0],
      "question":question,
      #"description": desc,
  })
  # ovde mi fali neka provera da li opis ima veze sa pitanjem ako ima onda ga vrati, ako nema onda kazi idk
  CURRENT_TEXT = answer
  return {"answer":answer, "image":get_image(metadata)}




graph_builder = StateGraph(State)

graph_builder.add_node("intent_classifier", classify_intent)
graph_builder.add_node("formulas_index", formulas_retriever)
graph_builder.add_node("images_index",images_retriever)
graph_builder.add_node("chroma_index",text_retriever)
graph_builder.add_node("formula_context",search_for_formula_context)
graph_builder.add_node("image_context_search",search_for_image_context)
graph_builder.add_node("show_image",show_image)

graph_builder.add_edge(START, "intent_classifier")
graph_builder.add_conditional_edges(
    "intent_classifier",
    decide_next_intent,
    {
        "formulas_index":"formulas_index",
        "images_index":"images_index",
        "chroma_index":"chroma_index",
    },
)
graph_builder.add_edge("formulas_index", "formula_context")
graph_builder.add_edge("formula_context",END)
graph_builder.add_edge("images_index","image_context_search")
graph_builder.add_edge("image_context_search","show_image")
graph_builder.add_edge("show_image",END)
graph_builder.add_edge("chroma_index",END)

graph = graph_builder.compile()


#######################FORMATIRANJE ODGOVORA##############################################################################

def format_formula(formula):
  if formula == None:
    return None 
  markdown = f"""$$
  {formula}
  $$"""
  return gr.Markdown(markdown, latex_delimiters=[{ "left": "$$", "right": "$$", "display": True },{"left":"\\[", "right":"\\]", "display":True}],)


# transformacija tekstualnog dela: 
def format_text(text):
  with gr.Blocks() as demo:
    return gr.Markdown(value=text,
              latex_delimiters=[
                  {"left": "\\(", "right": "\\)", "display": False},
                  {"left":"\\[", "right":"\\]", "display":True}, 
              ])

def format_image(io_bytes) -> Image.Image:
    if io_bytes == None:
      return None
    else:
      return Image.open(io_bytes)

#######################################################################################################################################




def ask_question(question):
  global CURRENT_FORMULA, CURRENT_IMAGE, CURRENT_TEXT
  inputs = {"question":question}

  for output in graph.stream(inputs):
    for key, value in output.items():
      pprint(f"Node '{key}':")
    pprint("\n --- \n")
  pprint(value["answer"])
  formula_to_return = CURRENT_FORMULA 
  image_to_return = CURRENT_IMAGE
  text_to_return = CURRENT_TEXT
  CURRENT_IMAGE = None 
  CURRENT_FORMULA = None
  CURRENT_TEXT = None 
  return format_text(text_to_return), format_image(image_to_return), format_formula(formula_to_return) 


if __name__ == "__main__":
  print("Sve je dobro importovano!")
  