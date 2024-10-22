from unstructured.chunking.basic import chunk_elements  
from unstructured.chunking.title import chunk_by_title 
from bert_for_text_classification import load_model, predict_probabilities 
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_community.embeddings import HuggingFaceEmbeddings
import aspose.words as aw 
import os 
import pdfminer 
#from pdfminer.utils import  open_filename
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json 
import pytesseract 
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR' # dodajem tesseract u Path varijable 
#C:\Program Files\Tesseract-OCR

def parse_pdf():
    # ekstrakcija teksta 
    elements = partition_pdf(
        filename='../data/ISLP.pdf',  # putanja do pdf fajla 
        strategy="hi_res", # strategija za obradu pdf dokumenta 
        infer_table_structure=True, # prepoznaje da u dokumentu postoji tabela i strukturira je 
        model_name="yolox", # model koji ćemo koristiti za prepoznavanje i analizu objekata u slikama 
        extract_images_in_pdf=True, 
        )

    
    #brisanje headera i footera 
    model, tokenizer =load_model()
    for element in elements:
        if str(element.__class__.__name__)=='Header' or str(element.__class__.__name__)=='Footer' or str(element.__class__.__name__)=='ListItem':
            probs = predict_probabilities(element.text,model, tokenizer)
        if probs[0][1].float() > 0.5: # znaci da je u pitanju junk
            elements.remove(element) # znaci da je pobrkao title sa headerom
    
   
    #chunkovanje 
    elements_chunk_by_title = chunk_by_title(elements,
                                            combine_text_under_n_chars=1000, # svi manji chunkovi od 7500 karaktera se kombinuju
                                            max_characters=1600, # maximalna velicina chunka
                                            multipage_sections=True,
                                            new_after_n_chars=1200, # ako je chunk veci od ovoga nece vise rasti do 8000
                                            overlap=True,
                                            #ovde je text separator by title, tj. \n\n, samim tim ne postoji ova opcija da se doda 
                                            )
    return elements_chunk_by_title