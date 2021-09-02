from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from torch import nn
import torch
class t5(nn.Module):
    def __init__(self,checkpoint:str):
        super(t5,self).__init__()
        self.config=T5Config.from_pretrained(checkpoint)
        self.t5=T5ForConditionalGeneration.from_pretrained(checkpoint,config=self.config)       
        
    
    def forward(self,input_ids,attention_mask,labels,label):
        output=self.t5(input_ids=input_ids,labels=labels,attention_mask=attention_mask,return_dict=True)       
        logits=output.logits
        batch_score=logits[:,0,[6136,1176]]
        #<6136 false> <1176 true>
        return batch_score
    
