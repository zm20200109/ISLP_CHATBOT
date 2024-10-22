import pickle 
from pydantic import BaseModel 
from typing import Any 
from pydantic import Field 
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import AIMessage 
import io 
import base64 
from PIL import Image 
from langchain_core.messages import HumanMessage
import os 
import uuid 
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.storage import InMemoryStore 
from langchain_core.documents import Document 
from langchain_chroma import Chroma 
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini",temperature=0)
gpt_vision_llm = ChatOpenAI(model="gpt-4o",openai_api_key=OPENAI_API_KEY) # gpt-4-vision-preview je zamenjen gpt-4o modelom 


class Element(BaseModel):
    type:str = Field(description="Type of pdf documents' element.")
    text:Any = Field(description="This is text content of pdf document's element")    
    page_no:int = Field(description="Page number of the original document from where chunk belongs.")

def load_text_summarize_chain():

    def parse(ai_message: AIMessage) -> str:
        """Parse the AI message."""
        return ai_message.content
    
    prompt_text = """ You are an assistant who's task is to summarize tables and chunks made of text. \
    Retrieve the summary response in the following format: ''
    Give precise and informative summary of table or text. Table or text element \
    is provided to you from following content: {element} 
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element":lambda x: x} | prompt | chat | parse 
    return summarize_chain 

def summarize_texts(text_elements):
    """
    text_elements: List[Element]
    returns: text_originals - list[str], text_summaries - list[str]
    """
    summarize_chain = load_text_summarize_chain() # chain za sumarizaciju 
    text_originals = [element.text for element in text_elements]  
    text_summaries = summarize_chain.batch(text_originals,{"max_concurrency":5})
    return text_originals, text_summaries  

def summarize_tables(table_elements):
    """
    table_elements: List[Element] where element.text represents html tags 
    returns: table_originals - list[str], text_originals - list[str]
    """
    summarize_chain = load_text_summarize_chain()
    table_originals = [table.text for table in table_elements] 
    table_summaries = summarize_chain.batch(table_originals,{"max_concurrency":5})
    return table_originals, table_summaries


def load_data(fpath):
    with open(fpath,'rb') as file:
        chunks = pickle.load(file)
        print(f"{len(chunks)} loaded from {fpath}.")
    return chunks 

##########################################################IMAGE SUMMARY CHAIN##################################################################################################
def image_to_base64(img_path):
    with Image.open(img_path) as image:
        buffered = io.BytesIO()
        image.save(buffered,format=image.format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')
    
def get_jpg_files(folder_path):
    jpg_files = [] 
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root,file))
    return jpg_files

def get_decoded_representations():
    representations = list()
    folder_path = "figures/"
    jpg_files = get_jpg_files(folder_path)
    for jpg_path in jpg_files:
        print(jpg_path)
        representations.append((jpg_path,image_to_base64(jpg_path)))
    return representations

def create_image_summaries():
    representations = get_decoded_representations()
    image_summaries = [] 
    for representation in representations:
        summary = gpt_vision_llm.invoke([
        HumanMessage(
            content=[
                {"type":"text","text":"Please give me a summary of the image provided. Be descriptive."},
                {
                    "type":"image_url",
                    "image_url":{
                    "url":f"data:image/jpeg;base64,{representation[1]}"
                    }
                }
                ]
            )
        ])
        image_summaries.append((representation[0],summary.content))
        with open("results/image_summaries.pkl","wb") as file2:
            pickle.dump(image_summaries,file2)
    return image_summaries

###########################MULTIVECTOR RETRIEVER + CHROMA VECTORSTORE#######################################################################################################################

def create_chroma_vectorstore(table_originals,text_originals,table_summaries,text_summaries,image_summaries,formula_descriptions,search_type:str,k:int):

    vectorstore = Chroma(
        collection_name="summaries",embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    )
    #2. store za originalne dokumente 
    store = InMemoryStore()
    id_key="doc_id"

    if search_type=="similarity":
        s_type = SearchType.similarity
    else:
        s_type=SearchType.mmr 
    
    #3. MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_type=s_type, # similarity ili mmr 
        search_kwargs={"k":k} # bilo 15 
    )
    #delete_collection(retriever)

    #img_ids = [str(img[0]) for img in image_summaries]
    #summary_img = [Document(page_content=image_tpl[1], metadata={id_key:image_tpl[0]}) for image_tpl in image_summaries] # lista dokumenata 
    #retriever.vectorstore.add_documents(summary_img) 
    #retriever.docstore.mset(image_summaries) 
    
    table_ids = [str(uuid.uuid4()) for _ in table_originals] # napravimo id-jeve od originalnih tabela 
    summary_tables = [Document(page_content=s, metadata={id_key:table_ids[i]}) for i, s in enumerate(table_summaries)] # te id-jeve prosledimo summaryjima i od njih napravimo documente i te dokumente ubacimo u vectorstore
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids,table_originals))) # originali u obliku ebcre27381:'html notacija za original tabelu u stringu'
     
    # tekstovi: text_originals, text_summaries 
    text_ids = [str(uuid.uuid4()) for _ in text_originals]
    summary_texts = [Document(page_content=s, metadata={id_key:text_ids[i]}) for i,s in enumerate(text_summaries)]
    retriever.vectorstore.add_documents(summary_texts) # dokumenti svih summaryja sa idjevima koji odgovaraju originalima 
    retriever.docstore.mset(list(zip(text_ids,text_originals)))

    # dodavanje dormula: 
    #formulas_documents = [Document(page_content=formula[1],metadata={id_key:formula[0]}) for formula in formula_descriptions]
    #retriever.vectorstore.add_documents(formulas_documents)
    #retriever.docstore.mset(formula_descriptions)

    print(f"{len(retriever.vectorstore.get()['documents'])} documents stored.")

    return retriever 

def delete_collection(retriever):
    # ovako brises kolekciju 
    retriever.vectorstore.delete_collection()
    # ovako mozes obrisati ceo docstore
    keys = retriever.docstore.store.keys()
    retriever.docstore.mdelete(keys=list(keys))


def create_retriever(chunks_path,txt_summaries_path,table_summaries_path,search_type,k):
    """
    search type - similariry, mmr
    k - 5,6,10
    """
    raw_pdf_elements = load_data(chunks_path)
    table_elements = list()
    text_elements = list()

    for element in raw_pdf_elements:
        if "CompositeElement" == element.category:
            text_elements.append(Element(type="text",text=element.text,page_no=dict(element.metadata.fields)['page_number']))
        elif "Table" == element.category:
            table_elements.append(Element(type="table",text=element.metadata.text_as_html,page_no=dict(element.metadata.fields)['page_number'])) # metadata.text_as_html

    # sumarizacija teksta, tabela, slika 
    table_originals = [table.text for table in table_elements] 
    text_originals = [element.text for element in text_elements]
    table_summaries = load_data(table_summaries_path)
    txt_summaries = load_data(txt_summaries_path)
    image_summaries = load_data('data_dir/1_image_summaries.pkl')
    formula_descriptions = load_data('data_dir/formula_descriptions.pkl')
    retriever = create_chroma_vectorstore(table_originals,text_originals,table_summaries,txt_summaries,image_summaries,formula_descriptions,search_type,k)
    return retriever 


if __name__=="__main__":
    # chunks_path = "/content/drive/MyDrive/project_work/code/results/chunked_elements_1.pkl"#"results/chunked_elements_1.pkl" # chunk_by_title strategy, combine_text_under_n_chars=1000,max_characters=1600, multipage_sections=True, new_after_n_chars=1200, overlap=True,
    # txt_summaries_path = "/content/drive/MyDrive/project_work/code/results/1_txt_summaries.pkl"#"results/1_txt_summaries.pkl"
    # tbl_summaries_path = "/content/drive/MyDrive/project_work/code/results/1_table_summaries.pkl"#"results/1_table_summaries.pkl"
    # retriever = create_retriever(chunks_path,txt_summaries_path,tbl_summaries_path,"similarity",5)
    pass 