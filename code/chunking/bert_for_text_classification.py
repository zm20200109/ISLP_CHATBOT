from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments 
from sklearn.model_selection import train_test_split 
import torch 
import pandas as pd 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import accelerate 
import transformers 
import numpy as np 
import csv 

class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels=None): 
        self.encodings = encodings
        self.labels = labels 

    def __getitem__(self, idx):
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item 
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    print({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
def predict_probabilities(input,model,tokenizer):
    text = input 
    inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    print(outputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions) # znaci INTRODUCTION pripada klasi no tj. nije header 
    return predictions

def train_and_save_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2)


    dataset = pd.read_csv("../data/junk_classification.csv")
    dataset = dataset.fillna('')
    X = list(dataset['text'])
    y = list(dataset['junk'])
    y = [1 if label == 'yes' else 0 for label in y]
    test_size = 0.3

    X_train, X_val, y_train,y_val = train_test_split(X,y,test_size=test_size,stratify=y)
    X_train_tokenized = tokenizer(X_train, padding=True,truncation=True,max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True,truncation=True,max_length=512)

    train_dataset = Dataset(X_train_tokenized,y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

     
    args = TrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        per_device_train_batch_size=8

    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )  
    trainer.train()
    trainer.evaluate()
    
    model_path = "../code/models/bert_model.pth" 
    torch.save(model.state_dict(), model_path)


def load_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2)
    model_path = '../code/models/bert_model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.eval(), tokenizer 

if __name__ == "__main__":
    #train_and_save_model() 
    model, tokenizer =load_model()
    predictions = predict_probabilities("1. Introduction 7",model, tokenizer) # 0 - no, 1 - yes 
    print("PREDICTIONS ", predictions)
    # 44 2. Statistical Learning 99.8% je junk, a 0.02% nije junk
    # 3.2.2 Some Important Questions. ............. 83 99.
    # 1. Many statistical learning methods are relevant and useful in a wide range of academic and non-academic disciplines, beyond just the sta- tistical sciences. We believe that many contemporary statistical learn- ing procedures should, and will, become as widely available and used as is currently the case for classical methods such as linear regres- sion. As a result, rather than attempting to consider every possible approach (an impossible task), we have concentrated on presenting he methods that we believe are most widely applicable. - 99.9% nije
    # 1. Introduction 7 - 99.65% jeste junk


