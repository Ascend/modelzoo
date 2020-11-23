 **介绍** 

Ascend ModelZoo，欢迎各位开发者

 **贡献要求** 

开发者提交的模型包括源码、readme、参考模型license文件、测试用例和readme，并遵循以下标准

 **一、源码** 

1、训练及在线推理请使用python代码实现，Ascend平台离线推理请使用C++代码，符合第四部分编码规范

2、参考[sample](https://gitee.com/ascend/modelzoo/tree/master/built-in/Official/nlp/Transformer_for_TensorFlow)

3、贡献者模型代码目录规则："modelzoo/contrib/Research/应用领域(nlp、cv、audio等)/网络名(全小写)/网络名_框架_华为云ID"（社区管理团队会在贡献完成进行整合）

 **二、readme** 

readme用于指导用户理解和部署样例，要包含如下内容：

- 简介：

1、模型的来源及原理；

2、模型复现的步骤，含训练、eval、在线/离线推理等，入口请封装成.sh、.py；

- 关键要求：

1、模型的出处、对数据的要求、免责声明等，开源代码文件修改需要增加版权说明；

2、模型转换得到的离线模型对输入数据的要求；

3、环境变量设置，依赖的第三方软件包和库，以及安装方法；

4、精度和性能达成要求：尽量达到原始模型水平；

5、数据集、预训练checkpoint、结果checkpoint请提供归档OBS、网盘链接，如来自开源需明确来源地址

 **三、自测试用例** 

提供模型的自测试用例和readme，提交PR需要门禁及模型测试用例通过，性能和精度检查通过

- 简介：

1、不同于完整的训练过程和全量数据集的推理，自测试用例的目的是验证提交代码基本功能可用，执行时长控制在10min之内(推理或训练只需执行有限的图片或step)；

2、提交PR中训练用例入口train_testcase.sh, 在线推理用例入口online_inference_testcase.sh, 离线推理用例入口offline_inference_testcase.sh；

3、提交PR后，会自动触发门禁流水，后台会根据用例入口shell，自动将代码分发到对应执行环境；

4、Jekins预置账号：global_read/huawei@123，登录之后，可以查看到用例执行日志

- 关键要求：

1、自测试用例命名严格按照上述简介2要求来书写，否则门禁会校验失败；

2、用例应当包含精度（Loss值）、性能检查，检查通过打印"Run testcase success!"，失败则打印"Run testcase failed!"；

3、执行环境已预装软件包和Python3.7.5环境，调用命令"python3"、"python3.7"、"python3.7.5"均可，安装第三方库依赖使用"pip3"、"pip3.7"均可；

4、数据集和模型：小于500M的文件，建议使用obsutil命令下载(已预装)，过大的文件，建议提交Issue，注明数据集和下载地址，会提前下载到执行环境上,

已预置数据集&python第三方库: [Environments](https://gitee.com/ascend/modelzoo/blob/master/contrib/ENVIRONMENTS.md)

5、环境和其他问题，请提交Issue跟踪；

6、测试用例开发参考：
[训练](https://gitee.com/ascend/modelzoo/tree/master/built-in/Official/nlp/Transformer_for_TensorFlow)
[离线推理](https://gitee.com/ascend/modelzoo/tree/master/contrib/Research/cv/efficientnet-b8/ATC_efficientnet-b8_tf_nkxiaolei)

 **四、PR提交**

- 关键要求：

1、请将modelzoo仓fork到个人分支,基于个人分支新增、修改和提交PR；

2、PR标题：线上活动，请在标题注明[线上贡献]；高校活动，请注明[xxx学校][高校贡献]；

 **五、编程规范** 

- 规范标准

1、C++代码遵循google编程规范：Google C++ Coding Guidelines；单元测测试遵循规范： Googletest Primer。

2、Python代码遵循PEP8规范：Python PEP 8 Coding Style；单元测试遵循规范： pytest

- 规范备注

1、优先使用string类型，避免使用char*；

2、禁止使用printf，一律使用cout；

3、内存管理尽量使用智能指针；

4、不准在函数里调用exit；

5、禁止使用IDE等工具自动生成代码；

6、控制第三方库依赖，如果引入第三方依赖，则需要提供第三方依赖安装和使用指导书；

7、一律使用英文注释，注释率30%--40%，鼓励自注释；

8、函数头必须有注释，说明函数作用，入参、出参；

9、统一错误码，通过错误码可以确认那个分支返回错误；

10、禁止出现打印一堆无影响的错误级别的日志；