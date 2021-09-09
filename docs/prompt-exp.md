# Prompt相关实验

现在我们需要研究Prompt在MS MARCO Passage数据集上的情况，需要比较BERT-Prompt，BERT以及T5在少量样本下的效果。

## 数据准备

使用MS MARCO Passage数据集。在a100机器下的这个文件夹：

```
/data/private/yushi/collections/msmarco-passage
```

里面有下列文件：

```
collection.tsv         
qrels.dev.small.tsv    
qrels.train.tsv        
queries.dev.small.tsv  
queries.dev.tsv        
queries.eval.small.tsv 
queries.eval.tsv       
queries.train.tsv      
```

`collection.tsv`是文档集合。其他的有dev和eval集的query以及qrels标注。用`head`查看文件格式。用带`small`版本的，不用不带`small`的，太大。eval文件可以先无视掉，其标注是不公开的。

首先我们需要组织成OpenMatch可以接受的格式，包含一个训练文件一个dev文件。OpenMatch接受的训练文件以及dev文件的格式，看我之前跟你说的robust04那个folder里面，有例子。统一使用classification，不使用其他格式的文件！对于train文件，正例直接从`qrels.train.tsv`中选取，训练集中的每个query选取1正例即可，如果某个query不含标注就扔掉。负例从BM25检索列表中选取。我已经在train和dev集合上做了一遍bm25检索，位置在`/data/private/yushi/runs`。下面的文件带train的就是在训练集上检索的。文件格式就是标准的trec检索列表格式，之前给你看过的。准备训练文件时，打开训练集上的检索列表，里面的检索结果排除掉qrels中标注的正例就当作负例（当然，这样会有false negative的情况，但这是通用做法）。准备dev文件时，直接转换dev上检索列表的格式至OpenMatch接受格式即可，不必关心正例是否存在。

## 实验目标

选取少量样本，我目前打算是从训练集中选取5个训练query，每个query带一个相关文档（MS MARCO数据集平均每个query确实也就1个相关文档标注），然后每个(q, d+) pair带不等数量的无关文档(d-)，可以去1~100里面的若干几个值。先比较BERT (base), BERT-Prompt (base), T5 (base)几个模型的情况。现在缺T5的实现，请你参考它的论文和代码在OpenMatch里实现一个T5 ranker。你大概需要学会huggingface transformers这个工具包，看官方文档和示例。

## 调用示例

展示OpenMatch训练与测试的示例命令，供参考，这些是我之前在robust04上做实验的命令

```
CUDA_VISIBLE_DEVICES=1,2,5,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345  train.py  -task classification  -model bert  -train ../ReInfoSelect_Testing_Data/robust04/fold_0/5q/rb04_train_classification.select0.neg20.jsonl  -max_input 1280000  -save ./checkpoints/bert-large-rb04-class-5q-select0.neg20.fold-0  -dev ../ReInfoSelect_Testing_Data/robust04/fold_0/rb04_dev.jsonl   -qrels ../ReInfoSelect_Testing_Data/robust04/rb04_qrels   -vocab ../pretrained_models/bert-large-uncased/          -pretrain ../pretrained_models/bert-large-uncased/  -res ./results/bert-large-rb04-class-5q-select0.neg20.fold-0.trec   -metric ndcg_cut_20  -max_query_len 20  -max_doc_len 489  -epoch 100  -batch_size 4  -lr 2e-5  -eval_every 50  -optimizer adamw  -dev_eval_batch_size 64  -n_warmup_steps 30  -logging_step 10  --log_dir=logs/bert-large-rb04-class-5q-select0.neg20.fold-0  --max_steps=5000
```

`-vocab`和`-pretrain`是用下载好的预训练模型，我放在了这里`/data/private/yushi/pretrained_models`，直接使用就可以。不过没有T5，需要你从huggingface页面上下载。校内机器下载会快一些。`-batch_size`是每个卡上的batch size。上面的命令会每隔1000步在dev集上做rerank。你需要根据实际情况调节各种参数，尤其是各种命名、warmup step数、eval step间隔（可能要调到几十）、max steps（最大步数）。这个命令里4张卡，每个batch size是8，总体是32。小样本学习的时候可能不需要开这么大，你看情况，但最好是固定一个。`log_dir`是tensorboard位置，vscode支持直接显示tensorboard。

上面的命令展示了使用普通的classification方法训练BERT ranker。要训练BERT-prompt，参考以下命令：

```
CUDA_VISIBLE_DEVICES=0,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345   train.py  -task prompt_classification  -model bert  -train ../ReInfoSelect_Testing_Data/robust04/fold_0/5q/rb04_train_classification.select0.neg20.jsonl  -max_input 1280000  -save ./checkpoints/bert-large-rb04-prompt-5q-select0.neg20.fold-0  -dev ../ReInfoSelect_Testing_Data/robust04/fold_0/rb04_dev.jsonl   -qrels ../ReInfoSelect_Testing_Data/robust04/rb04_qrels   -vocab ../pretrained_models/bert-large-uncased/          -pretrain ../pretrained_models/bert-large-uncased/  -res ./results/bert-large-rb04-prompt-5q-select0.neg20.fold-0.trec    -metric ndcg_cut_20  -max_query_len 20  -max_doc_len 489  -epoch 100  -batch_size 4  -lr 1e-5  -eval_every 50  -optimizer adamw  -dev_eval_batch_size 64  -n_warmup_steps 30  -logging_step 10  --max_steps=5000  
```

`-task`要改成`prompt_classification`，我还设置了一些命令行参数`--template`，`--pos_word`，`--neg_word`，打开源码你就能看到是什么意思了，可以修改模板以及label words。

以上命令Inference都是隔一定step就会自己做，你需要改相关参数，包括`-metric`，应当改成`mrr_cut_10`。`inference.py`里面的代码还未改可能不兼容。

在训练MS MARCO时，可能会遇到dev集太多Inference一次时间太长的情况。如果遇到这种情况，就重新构建一下dev集，从里面取样1/10的query即可。