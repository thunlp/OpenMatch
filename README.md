# OpenMatch v2

An all-in-one toolkit for information retrieval. Under active development.

# Install

```bash
git clone https://github.com/thunlp/OpenMatch.git
cd OpenMatch
pip install -e .
```

`-e` means **editable**, i.e. you can change the code directly in your directory.

We do not include all the requirements in the package. You may need to manually install `torch`, `tensorboard`.

You may also need faiss for dense retrieval. You can install either `faiss-cpu` or `faiss-gpu`, according to your enviroment. Note that if you want to perform search on GPUs, you need to install the version of `faiss-gpu` compatible with your CUDA. In some cases (usually CUDA >= 11.0) `pip` installs a wrong version. If you encounter errors during search on GPUs, you may try installing it from `conda`. 

# Features

- Human-friendly interface for dense retriever and re-ranker training and testing
- Various PLMs supported (BERT, RoBERTa, T5...)
- Native support for common IR & QA Datasets (MS MARCO, NQ, KILT, BEIR, ...)
- Deep integration with Huggingface Transformers and Datasets
- Efficient training and inference via stream-style data loading

# Docs

See docs folder.

# Project Organizers

- Zhiyuan Liu
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/~lzy/)
- Zhenghao Liu
  * Northeastern University
  * [Homepage](https://edwardzh.github.io/)
- Chenyan Xiong
  * Microsoft Research AI
  * [Homepage](https://www.microsoft.com/en-us/research/people/cxiong/)
- Maosong Sun
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/staff/sms/)

# Acknowledgments

Our implementation uses [Tevatron](https://github.com/texttron/tevatron) as the starting point. We thank its authors for their contributions.

# Contact

Please email to yushi17@foxmail.com.
