import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, quadruplet_process_fn, triple2dual_process_fn
import numpy as np
import logging
logger = logging.getLogger(__name__)
import random

# code for alignment and uniformity regularization
def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

def lunif(x, t=5):
    '''pdist doc: https://www.kite.com/python/docs/torch.pdist'''
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True,
            alignment_weight=0.0,
            uniformity_weight=0.0):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)

        if alignment_weight <=0.0 and uniformity_weight <=0.0:
            loss = torch.mean(-1.0 * lsm[:, 0])
            return (loss,{"loss":loss})
        else:
            q2d_loss = torch.mean(-1.0 * lsm[:, 0])
            alignment_loss = lalign(q_embs,a_embs)
            uniformity_loss = torch.mean(lunif(q_embs) + lunif(a_embs))
            loss = q2d_loss + alignment_weight * alignment_loss + uniformity_weight * uniformity_loss
            return (loss,{"loss":q2d_loss, "loss_align": alignment_loss, "loss_unif": uniformity_loss},)

class NLL_dual(EmbeddingMixin):
    def forward(
        self,
        query_ids,
        attention_mask_q,
        input_ids_a=None,
        attention_mask_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        neg_query_ids=None,
        attention_mask_neg_query=None,
        prime_loss_weight=1.0,
        dual_loss_weight=0.1,
        is_query=True):

        if neg_query_ids is None:
            return NLL.forward(self,query_ids,attention_mask_q,
            input_ids_a,attention_mask_a,input_ids_b,attention_mask_b,is_query=is_query)

        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        neg_q_embs = self.query_emb(neg_query_ids, attention_mask_neg_query)

        logit_matrix_q2d = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1),], dim=1)  # [B, 2]
        lsm_q2d = F.log_softmax(logit_matrix_q2d, dim=1)
        logit_matrix_d2q = torch.cat([(a_embs * q_embs).sum(-1).unsqueeze(1),
                                  (a_embs * neg_q_embs).sum(-1).unsqueeze(1),], dim=1)  # [B, 2]
        lsm_d2q = F.log_softmax(logit_matrix_d2q, dim=1)                                  
        loss_q2d = -1.0 * lsm_q2d[:, 0]
        loss_d2q = -1.0 * lsm_d2q[:, 0]
        loss_q2d = loss_q2d.mean()
        loss_d2q = loss_d2q.mean()
        loss = prime_loss_weight * loss_q2d + dual_loss_weight * loss_d2q
        return (loss,{"loss": loss_q2d,"loss_dual": loss_d2q},)

class NLL_cosine_SimCLR(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            alignment_weight=0.0,
            uniformity_weight=0.0,
            temperature=1.0,
            is_query=True,):

        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)


        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        # switch dot product to cosine similarity
        logit_matrix = torch.cat([torch.div((q_embs * a_embs).sum(-1).unsqueeze(1),temperature),
                                  torch.div((q_embs * b_embs).sum(-1).unsqueeze(1),temperature)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        
        loss = torch.mean(-1.0 * lsm[:, 0])
        scalar_dict={"loss":loss}
          
        if alignment_weight > 0.0 or uniformity_weight > 0.0:
            alignment_loss = lalign(q_embs,a_embs)
            loss = loss + alignment_weight * alignment_loss
            scalar_dict["loss_align_q2d"] = alignment_loss
            
            uniformity_loss = torch.mean(lunif(q_embs) + lunif(a_embs))
            loss = loss + uniformity_weight * uniformity_loss
            scalar_dict["loss_unif_q2d"] = uniformity_loss
            
        return (loss,scalar_dict)

class NLL_cosine_SimCLR_dual(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            neg_query_ids=None,
            attention_mask_neg_query=None,
            alignment_weight=0.0,
            uniformity_weight=0.0,
            temperature=1.0,
            prime_loss_weight=1.0,
            dual_loss_weight=0.1,
            is_query=True):

        assert dual_loss_weight > 0

        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)


        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        neg_q_embs = self.query_emb(neg_query_ids, attention_mask_neg_query)

        # switch dot product to cosine similarity
        logit_matrix_q2d = torch.cat([torch.div((q_embs * a_embs).sum(-1).unsqueeze(1),temperature),
                                  torch.div((q_embs * b_embs).sum(-1).unsqueeze(1),temperature)], dim=1)  # [B, 2]
        lsm_q2d = F.log_softmax(logit_matrix_q2d, dim=1)
        
        logit_matrix_d2q = torch.cat([torch.div((a_embs * q_embs).sum(-1).unsqueeze(1),temperature),
                                  torch.div((a_embs * neg_q_embs).sum(-1).unsqueeze(1),temperature)], dim=1)  # [B, 2]
        lsm_d2q = F.log_softmax(logit_matrix_d2q, dim=1)

        scalar_dict={}
        
        loss_q2d = torch.mean(-1.0 * lsm_q2d[:, 0])
        loss_d2q = torch.mean(-1.0 * lsm_d2q[:, 0])
        scalar_dict["loss"] = loss_q2d
        scalar_dict["loss_dual"] = loss_d2q
        
        loss = prime_loss_weight * loss_q2d + dual_loss_weight * loss_d2q

        if alignment_weight > 0.0 or uniformity_weight > 0.0:
            alignment_loss = lalign(q_embs,a_embs)
            loss = loss + alignment_weight * alignment_loss
            scalar_dict["loss_align_q2d"] = alignment_loss
            
            uniformity_loss = torch.mean(lunif(q_embs) + lunif(a_embs))
            loss = loss + uniformity_weight * uniformity_loss
            scalar_dict["loss_unif_q2d"] = uniformity_loss

        return (loss, scalar_dict)


class RobertaDot_NLL_LN(NLL, NLL_dual, NLL_cosine_SimCLR, NLL_cosine_SimCLR_dual, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)

        self.apply(self._init_weights)
        self.sparse_attention_mask_query = None
        self.sparse_attention_mask_document = None

        self.is_representation_l2_normalization = False # switch for L2 normalization after output
        self.is_projection_l2_normalization = False # do l2 normalization on an extra non-linear projection layer

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        if self.is_representation_l2_normalization:
            query1 = F.normalize(self.norm(self.embeddingHead(full_emb)), p=2, dim=1)
        else:
            query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        if self.sparse_attention_mask_document is not None:
            attention_mask =  self.convert_to_3D_mask(attention_mask,query=False)
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask) 
        if self.is_representation_l2_normalization:
            query1 = F.normalize(self.embeddingHead(full_emb), p=2, dim=1)
        else:
            query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            neg_query_ids=None,
            attention_mask_neg_query=None,
            is_query=True,
            alignment_weight=0.0,
            uniformity_weight=0.0,
            temperature=1.0,
            loss_objective="dot_product",
            prime_loss_weight=1.0,
            dual_loss_weight=0.0):

        if loss_objective == "dot_product":
            if dual_loss_weight > 0:
                # no alignment & uniformity
                return NLL_dual.forward(self,query_ids,attention_mask_q,
                    input_ids_a,attention_mask_a,input_ids_b,attention_mask_b,
                    neg_query_ids,attention_mask_neg_query,
                    prime_loss_weight=prime_loss_weight,dual_loss_weight=dual_loss_weight,
                    is_query=is_query
                )
            else :
                return NLL.forward(self,query_ids,attention_mask_q,
                        input_ids_a,attention_mask_a,input_ids_b,attention_mask_b,
                        is_query=is_query,
                        alignment_weight=alignment_weight,uniformity_weight=uniformity_weight
                )
        elif loss_objective == "simclr_cosine" :
            # with l2norm and temperature scaling
            if dual_loss_weight > 0:
                return NLL_cosine_SimCLR_dual.forward(self,
                    query_ids=query_ids,attention_mask_q=attention_mask_q,
                    input_ids_a=input_ids_a,attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,attention_mask_b=attention_mask_b,
                    neg_query_ids=neg_query_ids,attention_mask_neg_query=attention_mask_neg_query,
                    alignment_weight=alignment_weight,uniformity_weight=uniformity_weight,
                    temperature=temperature,
                    prime_loss_weight=prime_loss_weight,dual_loss_weight=dual_loss_weight,
                    is_query=is_query,extra_projection_loss=self.is_projection_l2_normalization
                )
            else:
                return NLL_cosine_SimCLR.forward(self,
                    query_ids=query_ids,attention_mask_q=attention_mask_q,
                    input_ids_a=input_ids_a,attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,attention_mask_b=attention_mask_b,
                    alignment_weight=alignment_weight,uniformity_weight=uniformity_weight,
                    temperature=temperature,
                    is_query=is_query,extra_projection_loss=self.is_projection_l2_normalization)

        else:
            logging.error("wrong loss type")
            exit()       

class NLL_MultiChunk(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor] -> if 0, means the corresponding part is full of padding token, ignore it.
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), a_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), b_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim] (or, [batch*chunkfactor, base_len, dim])

        if self.is_representation_l2_normalization:
        #     query1 = F.normalize(self.embeddingHead(full_emb), p=2, dim=1)
            compressed_output_k = F.normalize(compressed_output_k[:, 0, :], p=2, dim=1)
        else:
        #     query1 = self.norm(self.embeddingHead(full_emb))
            compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]

    def forward(
        self,
        query_ids,
        attention_mask_q,
        input_ids_a=None,
        attention_mask_a=None,
        input_ids_b=None,
        attention_mask_b=None,            
        loss_objective="dot_product",
        is_query=True):

        if loss_objective == "dot_product":

            return NLL_MultiChunk.forward(self,query_ids,attention_mask_q,
                        input_ids_a,attention_mask_a,input_ids_b,attention_mask_b,
                        is_query=is_query)



class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        #cfg = BertConfig.from_pretrained("bert-base-uncased")
        cfg = BertConfig.from_pretrained("/data/private/liyizhi/bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        #return cls.from_pretrained("bert-base-uncased", config=cfg)
        return cls.from_pretrained("/data/private/liyizhi/bert-base-uncased", config=cfg)
    def forward(self, input_ids, attention_mask):
        hidden_states = None
        sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                         attention_mask=attention_mask)
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args)
        self.ctx_model = HFBertEncoder.init_encoder(args)
        self.is_representation_l2_normalization = False # switch for L2 normalization after output

    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(input_ids, attention_mask)
        if self.is_representation_l2_normalization:
            return F.normalize(pooled_output, p=2, dim=1)
        return pooled_output
    def body_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.ctx_model(input_ids, attention_mask)
        if self.is_representation_l2_normalization:
            return F.normalize(pooled_output, p=2, dim=1)
        return pooled_output
    def forward(
        self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None,
        neg_query_ids=None, attention_mask_neg_query=None,dual_loss_weight=0.0,temperature=1.0,
    ):
        if input_ids_b is None:
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)
            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        if self.is_representation_l2_normalization:
            logit_matrix = torch.cat([torch.div((q_embs*a_embs).sum(-1).unsqueeze(1),temperature), torch.div((q_embs*b_embs).sum(-1).unsqueeze(1),temperature)], dim=1) #[B, 2]
        else:
            logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = torch.mean(-1.0*lsm[:,0])

        scalar_dict = {}
        scalar_dict["loss"]=loss
        
        if neg_query_ids is not None and dual_loss_weight > 0:
            neg_q_embs = self.query_emb(neg_query_ids, attention_mask_neg_query)
            
            if self.is_representation_l2_normalization:
                logit_matrix_d2q = torch.cat([torch.div((a_embs * q_embs).sum(-1).unsqueeze(1),temperature),
                                            torch.div((a_embs * neg_q_embs).sum(-1).unsqueeze(1),temperature)], dim=1)  # [B, 2]
            else:
                logit_matrix_d2q = torch.cat([(a_embs * q_embs).sum(-1).unsqueeze(1),
                                            (a_embs * neg_q_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
            lsm_d2q = F.log_softmax(logit_matrix_d2q, dim=1)
            loss_d2q = torch.mean(-1.0 * lsm_d2q[:, 0])
            scalar_dict["loss_dual"] = dual_loss_weight * loss_d2q
            loss = loss + dual_loss_weight * loss_d2q

        return (loss,scalar_dict)
        


# --------------------------------------------------
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            RobertaConfig,
        )
    ),
    (),
)


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig,dual_training_fn=False):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class

        if dual_training_fn:
            self.process_fn = quadruplet_process_fn


configs = [
    MSMarcoConfig(name="rdot_nll",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                ),
    MSMarcoConfig(name="rdot_nll_multi_chunk",
                model=RobertaDot_CLF_ANN_NLL_MultiChunk,
                use_mean=False,
                ),
    MSMarcoConfig(name="dpr",
                model=BiEncoder,
                tokenizer_class=BertTokenizer,
                config_class=BertConfig,
                use_mean=False,
                ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
