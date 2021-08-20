mkdir ../data/raw_data/
cd ../data/raw_data/

# # download MSMARCO passage data
# wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
# tar -zxvf collectionandqueries.tar.gz
# rm collectionandqueries.tar.gz

# wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz
# gunzip msmarco-passagetest2019-top1000.tsv.gz

# wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
# tar -zxvf top1000.dev.tar.gz
# rm top1000.dev.tar.gz

# wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
# tar -zxvf triples.train.small.tar.gz
# rm triples.train.small.tar.gz

# download MSMARCO doc data
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
gunzip msmarco-doctrain-queries.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
gunzip msmarco-doctrain-qrels.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz

wget https://trec.nist.gov/data/deep/2019qrels-docs.txt

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz
gunzip msmarco-doctest2019-top100.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz
gunzip msmarco-docdev-top100.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
gunzip msmarco-docdev-queries.tsv.gz


# # clone DPR repo and download NQ and TriviaQA datasets
# cd ../../../
# git clone https://github.com/facebookresearch/DPR
# cd DPR
# python data/download_data.py --resource data.wikipedia_split.psgs_w100
# python data/download_data.py --resource data.retriever.nq
# python data/download_data.py --resource data.retriever.trivia
# python data/download_data.py --resource data.retriever.qas.nq
# python data/download_data.py --resource data.retriever.qas.trivia
# python data/download_data.py --resource checkpoint.retriever.multiset.bert-base-encoder