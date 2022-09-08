FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
LABEL maintainer="Yizhi Li <yizhi.li@hotmail.com>"
USER root

# installing full CUDA toolkit
RUN apt update
RUN pip install --upgrade pip
#RUN apt install -y build-essential g++ llvm-9-dev git cmake wget
RUN apt install -y build-essential g++ git cmake wget
RUN conda install -y -c conda-forge cudatoolkit-dev
# setting environment variables
ENV CUDA_HOME "/opt/conda/pkgs/cuda-toolkit"
ENV CUDA_TOOLKIT_ROOT_DIR $CUDA_HOME
ENV LIBRARY_PATH "$CUDA_HOME/lib64:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH "$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV CFLAGS "-I$CUDA_HOME/include $CFLAGS"

# warning: no torch and torchvision in the requirements, need to install in advance
RUN wget https://raw.githubusercontent.com/thunlp/OpenMatch/master/retrievers/venv_ANCE.requirements
RUN pip install -r venv_ANCE.requirements
RUN pip install tensorflow

WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN python setup.py install --cpp_ext --cuda_ext
WORKDIR /workspace

RUN git clone https://github.com/microsoft/ANCE.git
WORKDIR /workspace/ANCE
RUN python setup.py install
WORKDIR /workspace

RUN git clone https://github.com/thunlp/OpenMatch.git
WORKDIR /workspace/OpenMatch
RUN python setup.py install
WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
