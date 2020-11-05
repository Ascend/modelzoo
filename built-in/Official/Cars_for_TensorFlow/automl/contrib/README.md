本目录用于放置新增算法和不对外公开算法，对于新增算法，需要满足如下要求：

1. 新增算法的文件放到 `./vega/algorithms` 目录下，参考[说明](./vega/algorithms/README.md)。
2. 新增网络的相关文件放到 `./vega/search_space` 目录下，参考[说明](./vega/search_space/README.md)。
3. 必须要提供算示例文件，放到 `./examples` 目录，参考[说明](./examples/README.md)
4. 必须要提供Benchmark配置，放到 `./benchmark` 目录。
5. 必须要提供中文算法文档，放到 `./docs/cn/algorithms` 和 `./docs/en/algorithms` 目录下，该目录下有模板供参考。
6. 必须要提供中英文示例说明文档，放到 `./docs/cn/user` 和 `./docs/en/user` 目录下，该目录下有文档供参考。
7. 若新增了数据集，放到 `./vega/datasets` 目录，参考[说明](./vega/datasets/README.md)。
8. 若依赖了新的开源软件，必须要在 `./deploy/install_dependencies.sh` 中新增该开源软件。在新增前，需要分析该开源软件的License是否满足要求。
9. **严禁从开源软件中直接拷贝代码**。

运行和调测`contrib`目录下的算法方法：

1. 设置环境变量`PYTHONPATH`。以下假设vega源代码的目录是 `/home_path/automl`。
   1. 若使用`pip`安装了Vega，则还需要设置：
      `export PYTHONPATH=/home_path/automl/contrib`
   2. 若未安装Vega，则需要设置：
      `export PYTHONPATH=/home_path/automl/contrib:/home_path/automl`
2. 如上配置后，即可在 `contrib/examples/` 目录下面执行示例代码：
   `python3 ./run_example.py ./new_algorithms_config`
