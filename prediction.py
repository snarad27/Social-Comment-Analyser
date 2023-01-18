
import torch
import tqdm 
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda
import pandas as pd
import streamlit as st

device = 'cuda' if cuda.is_available() else 'cpu'

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 10)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def prediction(model,df1):
    #Input to tokenizer -->
    tok_path = r"C:\Users\snara\Downloads\Project\vocab_distilbert_news.bin"
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(tok_path)
    df1=df1.dropna()
    df1=df1.reset_index(drop=True)
    fin_outputs=[]
    #Input to model -->
    model_path= r"C:\Users\snara\Downloads\Project\pytorch_distilbert_news.bin"
    model = DistilBERTClass()
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    for i in df1['comment']:
        inputs = tokenizer.encode_plus(i, return_tensors="pt", add_special_tokens=True)
        model.eval()
    
        with torch.no_grad():
            ids = inputs['input_ids'].to(device, dtype = torch.long)
            mask = inputs['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = inputs['token_type_ids'].to(device, dtype = torch.long)
                
            outputs = model(ids, mask,token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            torch.cuda.empty_cache()
    df3= pd.DataFrame(fin_outputs)
    df2=pd.DataFrame()
    f = lambda x: 1 if x>0.5 else 0
    df2=df3.applymap(f)
    df2.columns = ['sentiment'	,'respect',	'insult', 'humiliate',	'status',	'dehumanize',	'violence',	'genocide',	'attack_defend',	'hatespeech']
    result = pd.concat([df1, df2], axis=1, join='inner')
    result.drop(result.columns[result.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    
       

    return st.dataframe(result)