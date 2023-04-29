# hash:sha256:1925edaeca449d0eb93421c57bbd6c4e8b63571664532f3444b88c556a55c50e
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        keras==2.4.3 \
        matplotlib==3.5.1 \
        numpy==1.22.3 \
        pandas==1.4.2 \
        scikit-learn==1.0.2 \
        seaborn==0.12.2 \
        tensorflow==2.4.1 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    sklearn==0.0.post4
