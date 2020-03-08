### for web app and inference only

first build the docker image
```bash
docker build -f trt/Dockerfile -t asr .
```
download ngc clie
```
wget -O ngccli_reg_linux.zip https://ngc.nvidia.com/downloads/ngccli_reg_linux.zip && unzip -o ngccli_reg_linux.zip && chmod u+x ngc
ngc registry model download-version nvidia/bert_tf_v2_${MODEL}_${FT_PRECISION}_${SEQ_LEN}:2
```

### BERT
run command to build the container
```bash
scripts/build.sh
```

download the weights
```bash
scripts/download_model.sh
```
launch the container
```bash
scripts/launch.sh
```

build the TensorRT engine
```bash
python builder.py -m models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o large_384.engine -b 1 -s 384 -c models/fine-tuned/bert_tf_v2_large_fp16_384_v2/ -f
```
put the TensorRT engine file to 
```
bert_trt/resources/large_384.engine
```
### ASR

Need to download the pre-trained weights from NGC
```bash
./ngc registry model download-version nvidia/jasperpyt_fp16:1
```

build the ASR TensorRT model by following command
```python
python trt/perf.py --ckpt_path checkpoints/jasper_fp16.pt --wav=notebooks/example1.wav --model_toml=configs/jasper10x5dr_nomask.toml --make_onnx --onnx_path jasper.onnx --engine_path jasper.plan --dynamic_shape
```

put the TensorRT engine file to 
```
asr_trt/resources/jasper.plan
```

### Tacotron2
Need to download the pre-trained weights
```
ngc registry model download-version nvidia/tacotron2pyt_fp16:2
ngc registry model download-version nvidia/waveglow256pyt_fp16:1

```
build tr engines
```
python exports/export_tacotron2_onnx.py --tacotron2 tacotron2pyt_fp16_v2/nvidia_tacotron2pyt_fp16_20190427   waveglow256pyt_fp16_v1/nvidia_waveglow256pyt_fp16 -o output
python exports/export_waveglow_onnx.py --waveglow waveglow256pyt_fp16_v1/nvidia_waveglow256pyt_fp16 --wn-channels 256 -o output/
python trt/export_onnx2trt.py --encoder output/encoder.onnx --decoder output/decoder_iter.onnx --postnet output/postnet.onnx --waveglow output/waveglow.onnx -o output/ --fp16
```
