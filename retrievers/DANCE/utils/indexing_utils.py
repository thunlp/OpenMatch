import faiss
import numpy as np
import pickle
import os
import glob

def clean_faiss_gpu():
    ngpu = faiss.get_num_gpus()
    tempmem = 0
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)

def get_gpu_index(cpu_index):
    gpu_resources = []
    ngpu = faiss.get_num_gpus()
    tempmem = -1
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    def make_vres_vdev(i0=0, i1=-1):
        " return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        return vres, vdev

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True 
    gpu_vector_resources, gpu_devices_vector = make_vres_vdev(0, ngpu)
    gpu_index = faiss.index_cpu_to_gpu_multiple(gpu_vector_resources, gpu_devices_vector, cpu_index, co)
    return gpu_index


def document_split_faiss_index(logger,args,checkpoint_step,top_k_dev,top_k,dev_query_emb,train_query_emb,emb_prefix="passage_",two_query_set=True):
    logger.info(f"***** processing faiss indexing in split-mode *****")

    emb_file_pattern = os.path.join(args.output_dir,f'{emb_prefix}{checkpoint_step}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    logger.info(f"pattern {emb_file_pattern}\n file lists: {emb_file_lists}")

    # [[scores-nparray,scores-nparray..],[ANCE_ids-nparray,ANCE_ids-nparray,...]]
    merged_candidate_pair_dev = {"D":None,"I":None}
    merged_candidate_pair_train = {"D":None,"I":None}

    
    index_offset=0

    for emb_file in emb_file_lists:
        with open(emb_file,'rb') as handle:
            sub_passage_embedding = pickle.load(handle)
        # embid_file = emb_file.replace('emb_p','embid_p')
        # with open(embid_file,'rb') as handle:
        #     sub_passage_embedding2id = pickle.load(handle)
        logger.info(f"loaded {emb_file} embeddings")
        logger.info(f"sub_passage_embedding size {sub_passage_embedding.shape}")
        dim = sub_passage_embedding.shape[1]
        faiss.omp_set_num_threads(args.faiss_omp_num_threads)
        cpu_index = faiss.IndexFlatIP(dim)
        logger.info("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
        index = get_gpu_index(cpu_index) if args.gpu_index else cpu_index
        index.add(sub_passage_embedding)        

        D, dev_I = index.search(dev_query_emb, top_k_dev) # [n_dev,top]; [n_dev,top]
        dev_I = dev_I + index_offset
        if merged_candidate_pair_dev["D"] is None:
            merged_candidate_pair_dev["D"] = D
            merged_candidate_pair_dev["I"] = dev_I
        else:
            merged_candidate_pair_dev["D"] = np.concatenate([merged_candidate_pair_dev["D"],D],axis=1) # [n_dev,topk_dev *2]
            merged_candidate_pair_dev["I"] = np.concatenate([merged_candidate_pair_dev["I"],dev_I],axis=1) # [n_dev,topk_dev *2]
            sorted_ind = np.flip(np.argsort(merged_candidate_pair_dev["D"],axis=1),axis=1) # descent sort along topk_dev *2 scores for each row in n_dev
            merged_candidate_pair_dev["D"]=np.take_along_axis(merged_candidate_pair_dev["D"], sorted_ind, axis=1)[:,:top_k_dev] # [n_dev,topk_dev *2]
            merged_candidate_pair_dev["I"]=np.take_along_axis(merged_candidate_pair_dev["I"], sorted_ind, axis=1)[:,:top_k_dev] # [n_dev,topk_dev *2]
        
        if two_query_set:
            D, I = index.search(train_query_emb, top_k_dev) # [n_dev,top]; [n_dev,top]
            I = I + index_offset
            if merged_candidate_pair_train["D"] is None:
                merged_candidate_pair_train["D"] = D
                merged_candidate_pair_train["I"] = I
            else:
                merged_candidate_pair_train["D"] = np.concatenate([merged_candidate_pair_train["D"],D],axis=1) # [n_dev,topk_dev *2]
                merged_candidate_pair_train["I"] = np.concatenate([merged_candidate_pair_train["I"],I],axis=1) # [n_dev,topk_dev *2]
                sorted_ind = np.flip(np.argsort(merged_candidate_pair_train["D"],axis=1),axis=1) # descent sort along topk_dev *2 scores for each row in n_dev
                merged_candidate_pair_train["D"]=np.take_along_axis(merged_candidate_pair_train["D"], sorted_ind, axis=1)[:,:top_k] # [n_dev,topk_dev *2]
                merged_candidate_pair_train["I"]=np.take_along_axis(merged_candidate_pair_train["I"], sorted_ind, axis=1)[:,:top_k] # [n_dev,topk_dev *2]

        index_offset = index_offset + sub_passage_embedding.shape[0]
        index.reset()
        sub_passage_embedding=None
    
    return merged_candidate_pair_dev["D"],merged_candidate_pair_dev["I"],merged_candidate_pair_train["D"],merged_candidate_pair_train["I"]


def loading_possitive_document_embedding(logger,output_dir,checkpoint_step,possitive_training_passage_id_embidx,emb_prefix="passage_"):
    # output_dir == args.output_dir
    # possitive_training_passage_id_embidx.shape (n_positive,), range in [0,n_all_doc_embs]
    # for dual training in sperate mode
    emb_file_pattern = os.path.join(output_dir,f'{emb_prefix}{checkpoint_step}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    logger.info(f"pattern {emb_file_pattern}\n file lists: {emb_file_lists}")
    idx_lower_bound=0
    positive_embedding=np.ones([possitive_training_passage_id_embidx.shape[0],768],dtype=float)
    for emb_file in emb_file_lists:
        with open(emb_file,'rb') as handle:
            embedding=pickle.load(handle)
            dt=embedding.dtype
        idx_upper_bound = idx_lower_bound + embedding.shape[0]
        sub_relative_idx = np.intersect1d(np.where(possitive_training_passage_id_embidx>=idx_lower_bound),
                                np.where(possitive_training_passage_id_embidx<idx_upper_bound)) 
        
        positive_embedding[sub_relative_idx]=embedding[possitive_training_passage_id_embidx[sub_relative_idx] - idx_lower_bound]
        idx_lower_bound += embedding.shape[0]

    assert positive_embedding.shape[0] == possitive_training_passage_id_embidx.shape[0]
    return positive_embedding.astype(dt)