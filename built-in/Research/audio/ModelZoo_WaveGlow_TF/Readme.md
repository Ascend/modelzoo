## WaveGlow: a Flow-based Generative Network for Speech Synthesis
   WaveGlow: a flow-based network capable of generating high quality speech from mel-spectrograms. WaveGlow combines insights
from Glow and WaveNet in order to provide fast, efficient and high-quality
audio synthesis, without the need for auto-regression. WaveGlow is implemented
using only a single network, trained using only a single cost function:
maximizing the likelihood of the training data, which makes the training
procedure simple and stable.


## Requirements
1. download the project code and unzip the code.
2. Install third-party python libraries of requirements. 
   ```command
   pip3 install -r requirements.txt
   ```
3. The Ascend AI Processor is installed.
4. configure Ascend AI enviroment variables. as follow:
   ```command
    export install_path=/usr/local/Ascend/nnae/latest
    # driver包依赖
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置
    export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
    #fwkacllib 包依赖
    export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
    export PATH=${install_path}/fwkacllib/ccec_compiler/bin:{install_path}/fwkacllib/bin:$PATH
    #tfplugin 包依赖
    export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
    # opp包依赖
    export ASCEND_OPP_PATH=${install_path}/opp
   ```
 5. modify IP address configuration file "1p.json" and "8p.json"
 
## Train your own model
1. Download [LJSpeech-1.1 Data](https://keithito.com/LJ-Speech-Dataset). unzip the data into LJSpeech-1.1 dir
2. use "ljspeech_to_tfrecords.py" preprocess and converts the data into tfrecords format
   modify the parameter and run the file as follow ("wave_dir" is raw dataset dir,"tfrecords_dir" is preprocessed tfrecord data dir ):
   ```command
   python3 ljspeech_to_tfrecords.py --wave_dir ./LJSpeech-1.1/wavs   --tfrecords_dir  ./data/tfrecords 
   ```
3. Modify the hyperparameter in shell files(run_train_1p.sh, run_train_8p.sh) or configure file(params.py). And then train your WaveGlow networks

   ```command
   1 NPU:
        bash run_train_1p.sh
   8 NPU:
        bash run_train_8p.sh  
   ```
   
4. Modify the hyperparameter in shell file(run_inference.sh) ,then do inference with your network

   ```command
   bash run_inference.sh
   ```

## when set batch size is 12, speeds and performance as follow:
   1 NPU speeds：17.1 samples/sec；8 NPU speeds：141.3 samples/sec；performance：Loss：-5.90
## training result log
   ```commond  
   epoch 99 - step 137793 - loss = -5.999, lr=0.00010000, time cost=0.646627, samples per second=148.462672
   epoch 99 - step 137794 - loss = -6.008, lr=0.00010000, time cost=0.646930, samples per second=148.393075
   epoch 99 - step 137795 - loss = -5.850, lr=0.00010000, time cost=0.646943, samples per second=148.390286 
   epoch 99 - step 137796 - loss = -5.921, lr=0.00010000, time cost=0.646446, samples per second=148.504286 
   epoch 99 - step 137797 - loss = -5.940, lr=0.00010000, time cost=0.646566, samples per second=148.476687
   epoch 99 - step 137798 - loss = -5.915, lr=0.00010000, time cost=0.647016, samples per second=148.373554
   epoch 99 - step 137799 - loss = -5.849, lr=0.00010000, time cost=0.645823, samples per second=148.647594
  ```