# FAIS = Facebook AI Similarity Search
from uuid import uuid4 
from langchain_core.documents import Document
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
import faiss 
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS 
from mongo_db_connection import get_data, Element 
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.storage import InMemoryStore 
from dotenv import load_dotenv 
import os 

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 

def create_retriever():
    index = faiss.IndexFlatL2(len(embeddings.embed_query("Dimenzionalnost vektora je 1536."))) # FAISS index koji koristi euklidsku metruku za merenje sličnosti vektora. 
    vector_store = FAISS( # FAISS vectorstore cuva embeddinge i dokumente 
        embedding_function = embeddings,
        index=index, # FAISS index koji koristi Euklidsko L2 rastojanje za poredjenje vektora 
        docstore = InMemoryDocstore(), # mesto za čuvanje dokumenata čiji su vektori kreirani, InMemoryDocstore() znaci da se cuvaju u memoriji a ne na disku ili u bazi podataka
        index_to_docstore_id={}, # mapiranje izmedju FAISS indeksa i IDjeva dokumenata u docstore. Svaki vektor ima svoj identifikator koji se mapira na ID dokumenta u docstore-u. PRvo je mapiranje prazno, ai se popunjava tokom dodavanja dokumenata    
    )
    # u celini, ovaj deo je sistem koji FAISS koristi za pretragu izmedju sličnosti embedding vektora i povezivanje tih vektora sa dokumentima, koji su sačuvani u memoriji, skladište u FAISS INDEKS i pretražuju se koristeći L2 kao metriku sličnosti 
    id_key="doc_id"

    #summaryji za slike, tabele i tekstove
    img_summaries = get_data("results/1_image_summaries.pkl")
    img_ids = [str(img[0]) for img in img_summaries]
    summary_img = [Document(page_content=image_tpl[1], metadata={id_key:image_tpl[0]}) for image_tpl in img_summaries]
    vector_store.add_documents(documents=summary_img,ids=img_ids)


    tbl_summaries = get_data("results/1_table_summaries.pkl")
    tbl_originals = get_data("results/raw_table_elements.pkl")
    table_ids = [str(uuid4()) for _ in tbl_originals] # napravimo id-jeve od originalnih tabela 
    summary_tables = [Document(page_content=s, metadata={id_key:table_ids[i]}) for i, s in enumerate(tbl_summaries)] 
    vector_store.add_documents(documents=summary_tables,ids=table_ids)


    txt_originals = get_data("results/raw_txt_elements.pkl")
    txt_summaries = get_data("results/1_txt_summaries.pkl")
    text_ids = [str(uuid4()) for _ in txt_originals]
    summary_texts = [Document(page_content=s, metadata={id_key:text_ids[i]}) for i,s in enumerate(txt_summaries)]
    vector_store.add_documents(documents=summary_texts,ids=text_ids)

    # dodavanje formula i njihovih deskripcija u vector store 
    formula_descriptions = get_data('results/formula_descriptions.pkl')
    formulas_documents = [Document(page_content=formula[1],metadata={id_key:formula[0]}) for formula in formula_descriptions]
    vector_store.add_documents(formulas_documents)

    # documents su to od čega mora da se kreira Embedding, to su summary - ji.
    #njihovi idjevi, to cemo napraviti 

    store = InMemoryStore()
    id_key="doc_id"

    #3. MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vector_store, # summaryji i njihovi embeddinzi
        docstore=store, # parent dokuments tj. originali 
        id_key=id_key, # mapiranje docstore, vectorstore
        search_type=SearchType.similarity, # similarity ili mmr
        search_kwargs={"k":6}  
    )
    retriever.docstore.mset(list(zip(text_ids,txt_originals)))
    retriever.docstore.mset(list(zip(table_ids,tbl_originals)))
    retriever.docstore.mset(img_summaries)
    retriever.docstore.mset(formula_descriptions)

    return retriever 



if __name__=='__main__':
    query = "What is a studentized residual, and how is it used to identify outliers?"
    retriever = create_retriever()
    results = retriever.invoke(
       query
    )
    
    for res in results:
        print(res)
    
