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
        self.soft_prompt=soft_prompt
        #print(soft_sentence)
        if soft_prompt: 
            #self.soft_sentence="Judge the Relevance between Query and Docuemnt. " if soft_sentence == None else soft_sentence
            #print(self.soft_sentence)
            self.prefix_soft_index,self.infix_soft_index,self.suffix_soft_index=[3,27569,10],[11167,10],[31484,17,10,1]
            self.prefix_soft_embedding_layer=nn.Embedding(3,self.config.hidden_size)
            self.infix_soft_embedding_layer=nn.Embedding(2,self.config.hidden_size)
            self.suffix_soft_embedding_layer=nn.Embedding(4,self.config.hidden_size)
            #self.soft_index=self.tokenizer(self.soft_sentence)['input_ids'][:-1]
            #self.spnum=len(self.soft_index)
            #self.soft_embedding_layer=nn.Embedding(self.spnum,self.config.hidden_size)
            self.prefix_soft_embedding_layer.weight.data=torch.stack(
                [self.normal_embedding_layer.weight.data[i,:].clone().detach().requires_grad_(True) for i in self.prefix_soft_index]
                )
            self.infix_soft_embedding_layer.weight.data=torch.stack(
                [self.normal_embedding_layer.weight.data[i,:].clone().detach().requires_grad_(True) for i in self.infix_soft_index]
                )
            self.suffix_soft_embedding_layer.weight.data=torch.stack(
                [self.normal_embedding_layer.weight.data[i,:].clone().detach().requires_grad_(True) for i in self.suffix_soft_index]
                )
            self.prefix_soft_ids=torch.tensor(range(3))
            self.infix_soft_ids=torch.tensor(range(2))
            self.suffix_soft_ids=torch.tensor(range(4))
            for param in self.t5.parameters():
                param.requires_grad_(False)
        

    def forward(self,input_ids,attention_mask,query_ids,doc_ids,query_attention_mask,doc_attention_mask):
        batch_size=input_ids.shape[0]
        decoder_input_ids=torch.zeros(batch_size,1,dtype=int).to(input_ids.device)
        if self.soft_prompt:
            #print("good")
            prefix_soft_ids=torch.stack([self.prefix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            infix_soft_ids=torch.stack([self.infix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            suffix_soft_ids=torch.stack([self.suffix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            #print(self.prefix_soft_embedding_layer.weight.data.requires_grad)
            prefix_soft_embeddings=self.prefix_soft_embedding_layer(prefix_soft_ids)
            infix_soft_embeddings=self.infix_soft_embedding_layer(infix_soft_ids)
            suffix_soft_embeddings=self.suffix_soft_embedding_layer(suffix_soft_ids)
            #soft_embeddings=self.soft_embedding_layer(soft_ids)
            query_embeddings=self.normal_embedding_layer(query_ids)
            doc_embeddings=self.normal_embedding_layer(doc_ids)
            #start_embeddings=self.normal_embedding_layer(input_ids[:,0]).reshape(batch_size,1,-1)
            
            input_embeddings=torch.cat(
                [prefix_soft_embeddings,query_embeddings,infix_soft_embeddings,doc_embeddings,suffix_soft_embeddings],
                dim=1
                )
            #print(input_embeddings.require_grad)
            prefix_soft_attention_mask=torch.ones(batch_size,3).to(input_ids.device)
            infix_soft_attention_mask=torch.ones(batch_size,2).to(input_ids.device)
            suffix_soft_attention_mask=torch.ones(batch_size,4).to(input_ids.device)
            #normal_attention_mask=attention_mask[:,:-self.spnum]
            attention_mask=torch.cat(
                [prefix_soft_attention_mask,query_attention_mask,infix_soft_attention_mask,doc_attention_mask,suffix_soft_attention_mask],
                dim=1
                )
            output=self.t5(
                inputs_embeds=input_embeddings,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True
                )
        else:
            #print(self.t5.get_input_embeddings().weight.data.requires_grad)
            output=self.t5(input_ids=input_ids,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True)
        logits=output.logits
        batch_score=logits[:,0,[6136,1176]] 
        return batch_score

