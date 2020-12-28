# Document Retrieval
BM25 is following [anserini](https://github.com/castorini/anserini), and ANN is following [ANCE](https://github.com/microsoft/ANCE).

## BM25 Guide
### MS MARCO Doc Ranking Examples
First, get the [msmarco-docs.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz) and [msmarco-docdev-queries.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz), and preprocess it to jsonl format. *{'id': doc_id, 'contents': doc}* for each line, save it to *collections/msmarco/msmarco-docs.jsonl*.

Then build BM25 index and search:
```
./bm25_retriever/bin/IndexCollection -collection JsonCollection -input ./collections/msmarco -index index-msmarco -generator LuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRawDocs
./bm25_retriever/bin/SearchCollection -index index-msmarco -topicreader TsvString -topics msmarco-docdev-queries.tsv -bm25 -output msmarco-doc.txt
```

## ANCE Guide
The guides of ANCE training and inference are available at [ance](./openmatch_ance_retriver_readme.md).
