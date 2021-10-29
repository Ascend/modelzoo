# Try it

Try to run our pre-trained COCO Mask R-CNN [ckpt](https://download.mindspore.cn/model_zoo/r1.3/temp/maskrcnn_ascend_v130_coco2017_official_cv_bs2_bbox37.4_nsegm32.9/).

# Installing extra packages

Mask R-CNN requires a few extra packages.  We can install them now:

```
sudo apt-get install -y python-tk && \
pip install Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```
