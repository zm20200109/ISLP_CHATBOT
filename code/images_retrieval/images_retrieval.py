import ragatouille
from ragatouille import RAGPretrainedModel, RAGTrainer
import os 


def load_index():
  # ucitavanje modela 

  path_to_model_drive = '/content/drive/MyDrive/project_work/code/images_retrieval/ImagesModel7551'
  RAGDrive = RAGPretrainedModel.from_pretrained(path_to_model_drive)



  # kreiranje direktorijuma za model 
  os.makedirs('/content/.ragatouille/colbert/none/2024-10/08/07.09.01/checkpoints/colbert', exist_ok=True)


  # kopiranje fajlova modela bitnih za indeks 
  os.system('cp -r /content/drive/MyDrive/project_work/code/images_retrieval/ImagesModel7551/* /content/.ragatouille/colbert/none/2024-10/08/07.09.01/checkpoints/colbert')


  # ucitavanje indeksa 
  index_path = '/content/drive/MyDrive/project_work/code/images_retrieval/images-fine-tuned-colbert-index-530000-1'
  indexDrive = RAGDrive.from_index(index_path)

  indexDrive.search("") # pokretanje searchera za index po prvi put .. (treba 3 minuta)

  return indexDrive 

def load_model_and_index():
  RAG = RAGPretrainedModel.from_pretrained("./fine-tuned-colbert-images-2")
  os.makedirs('/content/.ragatouille/colbert/none/2024-10/08/07.09.01/checkpoints/colbert', exist_ok=True)
  os.system('cp -r /content/fine-tuned-colbert-images-2/* /content/.ragatouille/colbert/none/2024-10/08/07.09.01/checkpoints/colbert')
  ind_path = "/content/images-fine-tuned-colbert-index-530000-1/"
  indexDrive = RAG.from_index(ind_path)
  indexDrive.search("")
  return indexDrive 


  if __name__=="__main__":
    # load_index()
    pass 