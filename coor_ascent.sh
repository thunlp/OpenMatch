java -cp LeToR/RankLib-2.1-patched.jar ciir.umass.edu.features.FeatureManager -input features/bert_features -output features/ -k 2
java -jar LeToR/RankLib-2.1-patched.jar -train features/bert_features -ranker 4 -kcv 2 -kcvmd checkpoints/ -kcvmn ca -metric2t NDCG@20 -metric2T NDCG@20
java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f1.ca -rank features/f1.test.bert_features -score f1.score
java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f2.ca -rank features/f2.test.bert_features -score f2.score
python LeToR/gen_trec.py -dev data/dev_toy.tsv -res results/bert_ca.trec -k 2
rm f1.score
rm f2.score
