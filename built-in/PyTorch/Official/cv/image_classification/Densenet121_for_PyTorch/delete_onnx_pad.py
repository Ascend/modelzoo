import onnx

model = onnx.load("densenet121_cpu_2.onnx")

# 重连pad输入边到pad后面的节点
model.graph.node[51].input[0] = '776'
model.graph.node[141].input[0] = '866'
model.graph.node[315].input[0] = '1040'

# 删除pad节点
node_list = ['Pad_50', "Pad_140", "Pad_314", "Constant_49", "Constant_139", "Constant_313"]
max_idx = len(model.graph.node)
rm_cnt = 0
for i in range(len(model.graph.node)):
    if i < max_idx:
        n = model.graph.node[i - rm_cnt]
        if n.name in node_list:
            print("remove {} total {}".format(n.name, len(model.graph.node)))
            model.graph.node.remove(n)
            max_idx -= 1
            rm_cnt += 1

onnx.checker.check_model(model)

onnx.save(model, "densenet121_official_nopad.onnx")