from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from torch import nn
import torch
class t5(nn.Module):
    def __init__(self,checkpoint:str):
        super(t5,self).__init__()
        self.config=T5Config.from_pretrained(checkpoint)       
        self.t5=T5ForConditionalGeneration.from_pretrained(checkpoint,config=self.config)      
    
    def forward(self,input_ids,attention_mask):
        batch_size=input_ids.shape[0]
        decoder_input_ids=torch.zeros(batch_size,1,dtype=int).to(input_ids.device)
        output=self.t5(input_ids=input_ids,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True)
        logits=output.logits
        batch_score=logits[:,0,[6136,1176]] 
        return batch_score

