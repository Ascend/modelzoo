迁移[Albert](https://github.com/google-research/albert)到ascend910平台
使用的是albert_v2版本的预训练模型
|  | F1| EM |
| :-----| ----: | :----: |
| albert_base(Ascend) | 82.4| 79.4|
| albert_base(论文) | 82.1 | 79.3 |
| albert_large(Ascend) | 84.2 | 81.3 |
| albert_large(论文) | 84.9 | 81.8 |


训练和预测脚本:

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只训练则注释掉--do_train

只预测则注释掉--do_predict

输入文件需要建立squad_v2文件夹

对于albert_base需要建立albert_base_v2, output_base_v2文件夹

对于albert_large需要建立albert_base_v2, output_base_v2文件夹

上述文件夹均可从

[百度网盘](https://pan.baidu.com/s/1F_8A398wefDj9woOJ71MwQ)提取码: 7taq 下载
