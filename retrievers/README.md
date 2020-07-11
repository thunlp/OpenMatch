# Document Retrieval
BM25 is following [anserini](https://github.com/castorini/anserini), and ANN is following [ANCE](https://github.com/microsoft/ANCE).

## Run
Build BM25 index:
```
./bm25_retriever/bin/IndexCollection -collection JsonCollection -input {your collection} -index {index path} -generator LuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRawDocs >& {log file path}
```
Search by BM25:
```
./bm25_retriever/bin/SearchCollection -index {index path} -topicreader {topic format} -topics {topic path} -bm25 -output {result file path}
```

## Data Format
BM25 accept *jsonl* format, each line is like this: *{'id': str, 'contents': str}*. Detailed examples are available [here](https://github.com/castorini/anserini).
