# ![MindSpore Logo](https://www.mindspore.cn/static/img/logo_black.6a5c850d.png)

## 欢迎来到MindSpore ModelZoo

为了让开发者更好地体验MindSpore框架优势，我们将陆续增加更多的典型网络和相关预训练模型。如果您对ModelZoo有任何需求，请通过[Gitee](https://gitee.com/mindspore/mindspore/issues)或[MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html)与我们联系，我们将及时处理。

- 使用最新MindSpore API的SOTA模型

- MindSpore优势

- 官方维护和支持

## 目录

MindSpore的模型库，主要维护在[MindSpore仓库](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)

### 标准网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative）|
|:----  |:-------  |:----   |:----:    |:----:    |:----: |:----: |:----: |:----: |
|计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet)   |  ✅ |  ✅ |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnn_direction_model)  |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet100](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/densenet) |   |   |   |   | ✅ | ✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet121](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/densenet) |  ✅ |  ✅ |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/dpn) |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [EfficientNet-B0](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/efficientnet) |   |   |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogLeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/googlenet)   |  ✅     |  ✅ | ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv3) |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV4](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv4) |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet)    |  ✅ |  ✅ |  ✅ |  ✅ | ✅ | ✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet_quant)    |  ✅ |   |  ✅ |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV1](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv1)        |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2)        |  ✅ |  ✅ |  ✅ |  ✅ | ✅ |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2_quant)        |  ✅ |   |  ✅ |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv3)        |   |   |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [NASNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/nasnet) |   |   |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-18](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)   |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)   |  ✅ |  ✅ |  ✅ |  ✅ | ✅ |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet50_quant)   |  ✅ |   |   |   |  |  |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)        |  ✅ |  ✅ | ✅ |  ✅ |  |  |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNeXt50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnext)    |  ✅ |   | ✅ |  ✅ |  |  |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-ResNet50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)       |  ✅ | ✅ |  |  |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV1](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/shufflenetv1)        |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/shufflenetv2) |   |   |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  |[SqueezeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Tiny-DarkNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/tinydarknet)  |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/vgg16)  |  ✅ |  ✅ |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Xception](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/xception) |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CenterFace](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/centerface)     |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CTPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ctpn)     |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn)  |  ✅ |   |  ✅ |   |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Mask R-CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/maskrcnn)  |  ✅ |  ✅ |   |   |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  |[Mask R-CNN (MobileNetV1)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/maskrcnn_mobilenetv1)         |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [RetinaFace-ResNet50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/retinaface_resnet50)   |   |   |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)                   |  ✅ |   | ✅ | ✅ | ✅ |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-MobileNetV1-FPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)         |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-Resnet50-FPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)                   |  ✅ |   |  |  |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-VGG16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)                   |  ✅ |   |  |  |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WarpCTC](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc)                    |  ✅ |   |  ✅ |   |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-ResNet18](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18)   |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53)   |  ✅ |  ✅ |  ✅ |  ✅ |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53_quant)   |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  |[YOLOv4](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov4)         |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 文本检测（Text Detection）  | [DeepText](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeptext)                |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 文本检测（Text Detection）  | [PSENet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/psenet)                |  ✅ |  ✅ |   |   |  |  |
| 计算机视觉（CV） | 文本识别（Text Recognition）  | [CNN+CTC](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnnctc)                |  ✅ |  ✅ |   |   |  |  |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeepLabV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeplabv3)   |  ✅ |   |   |   | ✅ |  |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net2D (Medical)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet)   |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net3D (Medical)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet3d)   |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net++](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet)                |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  |[OpenPose](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/openpose)                |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  |[SimplePoseNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/simple_pose)                |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 光学字符识别（Optical Character Recognition）  |[CRNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/crnn)                |  ✅ |   |   |   |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert)  |  ✅ |  ✅ |  ✅ |  ✅ |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [FastText](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/fasttext)    |  ✅ |   |   |  |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GNMT v2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gnmt_v2)    |  ✅ |   |   |  |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GRU](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gru)            |  ✅ |   |   |  |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/mass)    |  ✅ |  ✅ |  ✅ |  ✅ |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [SentimentNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm)    |  ✅ |   |  ✅ |  ✅ | ✅ | ✅ |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/transformer)  |  ✅ |  ✅ |  ✅ |  ✅ |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TinyBERT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/tinybert)   |  ✅ |  ✅ |  ✅ |  |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TextCNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/textcnn)            |  ✅ |   |   |  |  |  |
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/deepfm)    |  ✅ |  ✅ |  ✅ | ✅| ✅ |  |
| 推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search, Ranking）  | [Wide&Deep](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep)      |  ✅ |  ✅ |  ✅ | ✅ |  |  |
| 推荐（Recommender） | 推荐系统（Recommender System）  | [NAML](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/naml)             |  ✅ |   |   |  |  |  |
| 推荐（Recommender） | 推荐系统（Recommender System）  | [NCF](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/ncf)    |  ✅ |   |  | |  |  |
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gcn)  |  ✅ |  ✅ |   |   |  |  |
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gat) |  ✅ |  ✅ |   |   |  |  |
| 图神经网络（GNN） | 推荐系统（Recommender System） | [BGCF](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) |  ✅ |   |   |   |  |  |

### 研究网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative） |
|:----  |:-------  |:----   |:----:    |:----:    |:----: |:----: |:----: |:----: |
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceAttributes](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceAttribute)     |  ✅ |  ✅ |   |   |  |  |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [FaceDetection](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceDetection)  |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceQualityAssessment](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceQualityAssessment)     |  ✅ |  ✅ |   |   |  |  |
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceRecognition](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceRecognition)     |  ✅ |   |   |   |  |  |
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceRecognitionForTracking](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceRecognitionForTracking)     |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-GhostNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/ssd_ghostnet)           |  ✅ |   |   |   |  |  |
| 计算机视觉（CV）  | 关键点检测（Key Point Detection）  | [CenterNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/centernet)          |  ✅ |   |  |   | ✅ |  |
| 计算机视觉（CV）  | 图像风格迁移（Image Style Transfer）  | [CycleGAN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/CycleGAN)       |       |   |  |  ✅ | ✅ |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [DS-CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/dscnn)          |  ✅ |  ✅ |   |  |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TextRCNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/textrcnn)    |  ✅ |   |   |  |  |  |
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TPRR](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/tprr)  |  ✅ |   |   |  |  |  |
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction） | [AutoDis](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/recommend/autodis)   |  ✅ |   |   |   |  |  |
|语音（Audio） | 音频标注（Audio Tagging）  | [FCN-4](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/audio/fcn-4)   |  ✅ |   |   |   |  |  |
|高性能计算（HPC） | 分子动力学（Molecular Dynamics）  |  [DeepPotentialH2O](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/hpc/molecular_dynamics)   |  ✅ | ✅|   |   |  |  |
|高性能计算（HPC） | 海洋模型（Ocean Model）  |  [GOMO](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/hpc/ocean_model)   |   |   |  ✅ |   |  |  |

## 免责声明

MindSpore仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在MindSpore中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过GitHub或Gitee与我们联系。非常感谢您对这个社区的理解和贡献。

MindSpore已获得Apache 2.0许可，请参见LICENSE文件。

## 许可证

[Apache 2.0许可证](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)