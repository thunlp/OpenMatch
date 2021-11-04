from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from torch import nn
import torch
class t5(nn.Module):
    def __init__(self,checkpoint:str):
        super(t5,self).__init__()
        self.config=T5Config.from_pretrained(checkpoint)       
        self.t5=T5ForConditionalGeneration.from_pretrained(checkpoint,config=self.config)      
    
    def forward(self,input_ids,attention_mask,labels,label,isv11):
        output=self.t5(input_ids=input_ids,decoder_input_ids=labels,attention_mask=attention_mask,return_dict=True)
        logits=output.logits
        if not isv11:
            batch_score=logits[:,0,[6136,1176]] 
        else:
            batch_score=logits[:,0,:] 
        return batch_score

