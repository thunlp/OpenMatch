from numpy import datetime_data
from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from torch import nn
import torch
class t5(nn.Module):
    def __init__(self,checkpoint:str,soft_prompt:bool,soft_sentence:int):
        super(t5,self).__init__()
        self.config=T5Config.from_pretrained(checkpoint)       
        self.t5=T5ForConditionalGeneration.from_pretrained(checkpoint,config=self.config)  
        self.tokenizer=T5Tokenizer.from_pretrained(checkpoint)
        self.soft_embedding_layer=None   
        self.normal_embedding_layer=self.t5.get_input_embeddings()
        if soft_prompt: 
            self.soft_sentence="Judge the Relevance between Query and Docuemnt. " if soft_sentence == None else soft_sentence
            self.soft_index=self.tokenizer(self.soft_sentence)['input_ids'][:-1]
            self.spnum=len(self.soft_index)
            self.soft_embedding_layer=nn.Embedding(self.spnum,self.config.hidden_size)
            self.soft_embedding_layer.weight.data=torch.stack(
                [self.normal_embedding_layer.weight.data[i,:].clone().detach().requires_grad_(True) for i in self.soft_index]
                )
            self.soft_ids=torch.tensor(range(self.spnum))
            for param in self.t5.parameters():
                param.requires_grad_(False)

    def forward(self,input_ids,attention_mask):
        batch_size=input_ids.shape[0]
        decoder_input_ids=torch.zeros(batch_size,1,dtype=int).to(input_ids.device)
        if not self.soft_embedding_layer== None:
            soft_ids=torch.stack([self.soft_ids for i in range(batch_size)]).to(input_ids.device)
            soft_embeddings=self.soft_embedding_layer(soft_ids)
            normal_embeddings=self.normal_embedding_layer(input_ids[:,1:-self.spnum])
            start_embeddings=self.normal_embedding_layer(input_ids[:,0]).reshape(batch_size,1,-1)
            
            input_embeddings=torch.cat([start_embeddings,soft_embeddings,normal_embeddings],dim=1)
            soft_attention_mask=torch.ones(batch_size,self.spnum).to(input_ids.device)
            normal_attention_mask=attention_mask[:,:-self.spnum]
            attention_mask=torch.cat([soft_attention_mask,normal_attention_mask],dim=1)
            output=self.t5(
                inputs_embeds=input_embeddings,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True
                )
        else:
            output=self.t5(input_ids=input_ids,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True)
        logits=output.logits
        batch_score=logits[:,0,[6136,1176]] 
        return batch_score

