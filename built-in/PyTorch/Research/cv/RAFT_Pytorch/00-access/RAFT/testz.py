import torch
import torch.nn as nn
import torch.nn.functional as F
torch.npu.set_device(4)

ic=2
input = torch.rand(1,ic,36,120).to(torch.float)
k1=3
k2=3

#cpu
output = F.unfold(input, [k1,k2], padding=1)
print(output.size())


# conv2d based unfold
output = output.npu()
w = torch.zeros(ic*k1*k2,ic,k1,k2).to(torch.float).npu()
for i in range(ic):
    for j in range(k1):
        for k in range(k2):
            w[i*k1*k2+j*k2+k,i,j,k]=1
output2 = torch.nn.functional.conv2d(input.npu(), w, padding=1).view(1,ic*k1*k2,-1)

#print(output2)
#print(output2.size())
print(output-output2)
print((output-output2).abs().sum())
print((output-output2).abs().mean())




#inp = torch.randn(1, 3, 10, 12)
#w = torch.randn(2, 3, 4, 5)
#inp_unf = torch.nn.functional.unfold(inp, (4, 5))
#print(inp_unf.size())



#npu
#input = input.npu()
#output = F.unfold(8 * input, [3,3], padding=1)

#print(output.size())
#print(output)



