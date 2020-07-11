from setuptools import setup, find_packages

install_requires = [
    'torch==1.4.0',
    'transformers==2.8.0',
    'faiss-cpu==1.6.3',
    'nltk==3.5',
    'pytrec_eval==0.4'
]

setup(
    name="OpenMatch",
    version="0.0.1",
    author="OpenMatch Authors",
    author_email='zkt18{at}mails.tsinghua.edu.cn',
    description="An Open Source Package for Information Retrieval",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6'
)
