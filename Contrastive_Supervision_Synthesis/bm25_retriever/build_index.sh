export dataset_name= ## you need to set this
export data_path=## you need to set this

./bin/IndexCollection -collection JsonCollection -input $data_path/corpus -index $dataset_name -generator LuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRawDocs
