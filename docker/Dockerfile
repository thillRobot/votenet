ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility

ENV VN_WS=/home/votenet_ws
RUN mkdir -p $VN_WS 

RUN apt update && apt upgrade -y && apt install -y vim git curl wget unzip python3-dev python3-pip
RUN apt install -y build-essential ninja-build cmake libopenblas-dev xterm xauth openssh-server tmux mate-desktop-environment-core

RUN pip install --upgrade pip && pip install wheel numpy torch matplotlib pandas test-common Pillow open3d h5py
RUN pip install opencv-python plyfile trimesh==2.35.39
RUN pip install 'networkx>=2.2,<2.3'
#RUN pip install tensorflow # runtime issues, check on this or just skip it

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

COPY ./votenet /votenet
RUN cd /votenet/pointnet2; python setup.py install

# startup routine in entrypoint script
COPY ./shared/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
