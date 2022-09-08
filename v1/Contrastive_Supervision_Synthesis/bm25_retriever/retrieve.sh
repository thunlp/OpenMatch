export dataset_name= ## you need to set this
export generator_folder=qg_t5-base ## qg_t5-small ; qg_t5-base
export data_path= ## you need to set this

./bin/SearchCollection -index $dataset_name -topicreader TsvString -topics $data_path/qid2query.tsv -bm25 -output $data_path/bm25_retrieval.trec
