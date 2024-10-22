from ragatouille import RAGPretrainedModel
import os 

def load_formulas_index():

  # ucitavanje modela
  path_to_model_drive = '/content/drive/MyDrive/project_work/code/formulas_retrieval/FormulasColBERTCheckpointsFinal/'
  RAGDrive = RAGPretrainedModel.from_pretrained(path_to_model_drive)
  
  
  # kreiranje direktorijuma za model
  os.makedirs('/content/.ragatouille/colbert/none/2024-10/16/08.25.43/checkpoints/colbert', exist_ok=True)


  # Kopiranje fajlova sa Google Drive-a u kreirani dir 
  os.system('cp -r /content/drive/MyDrive/project_work/code/formulas_retrieval/FormulasColBERTCheckpointsFinal/* /content/.ragatouille/colbert/none/2024-10/16/08.25.43/checkpoints/colbert')


  # ucitavamo index kada smo kreirali path za model 
  index_path = '/content/drive/MyDrive/project_work/code/formulas_retrieval/colbert-formulas-index-5/'
  indexDrive = RAGDrive.from_index(index_path)

  # loadujem searcher za index
  indexDrive.search("")
  return indexDrive

def load_formulas_model_and_index():
  path_to_model = "./fine-tuned-colbert-formulas"
  RAG = RAGPretrainedModel.from_pretrained(path_to_model)
  os.makedirs('/content/.ragatouille/colbert/none/2024-10/16/08.25.43/checkpoints/colbert', exist_ok=True)
  os.system('cp -r /content/fine-tuned-colbert-formulas/* /content/.ragatouille/colbert/none/2024-10/16/08.25.43/checkpoints/colbert')
  index_path = '/content/colbert-formulas-index-5/'
  index = RAG.from_index(index_path)
  index.search("")
  return index 

  if __name__ == '__main__':
    pass 