Installation
==================

From PyPI
--------------------------------

::

    pip install git+https://github.com/thunlp/OpenMatch.git



Using Git Repository
----------------------------

::

    git clone https://github.com/thunlp/OpenMatch.git
    cd OpenMatch
    python setup.py install


Using Docker
----------------------------

To build an OpenMatch docker image from Dockerfile

::

    docker build -t <image_name> .

To run your docker image just built above as a container

::

    docker run --gpus all --name=<container_name> -it -v /:/all/ --rm <image_name>:<TAG>