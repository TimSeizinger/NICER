# FROM pytorch/pytorch is also possible, which simply takes the "latest" tag
FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN conda install pip
RUN pip install sklearn gensim numpy pandas tqdm

# Run a test if the container has all dependencies
RUN python -c "import torch, gensim"