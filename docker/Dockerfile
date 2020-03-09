ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.01-py3
# ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.10-py3 
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y python3

WORKDIR /tmp/onnx-trt
# COPY trt/onnx-trt.patch .
RUN git clone https://github.com/onnx/onnx-tensorrt.git && cd onnx-tensorrt && git checkout 7.0 && git submodule update --init --recursive && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DGPU_ARCHS="60 70 75" && make -j16 && make install && mv -f /usr/lib/libnvonnx* /usr/lib/x86_64-linux-gnu/ && ldconfig


# Here's a good place to install pip reqs from JoC repo.
# At the same step, also install TRT pip reqs
WORKDIR /tmp/pipReqs
COPY requirements.txt /tmp/pipReqs/jocRequirements.txt
RUN pip install --disable-pip-version-check -U -r jocRequirements.txt
RUN pip install tensorflow==1.13.1
# && pip install horovod
RUN pip install cherrypy pudb
# WORKDIR /workspace/jasper
#COPY . .
#COPY --from=bert-tensorrt:latest /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so* /usr/lib/x86_64-linux-gnu/
#RUN ldconfig
# Build with extra flags to enable plugins available on SM 70/75
RUN git clone https://github.com/NVIDIA/TensorRT.git --recursive && cd TensorRT
RUN cd TensorRT && git checkout release/7.0 && git submodule update --init --recursive && mkdir build && cd build && cmake .. -DCMAKE_CUDA_FLAGS="--expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75" && make -j nvinfer_plugin
# Replace the default plugin library with the open-source plugins
RUN cp TensorRT/build/libnvinfer_plugin.so* /usr/lib/x86_64-linux-gnu/
# install the tokenizer
RUN python -m nltk.downloader 'punkt'
