import torch
import numpy
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray 
import torch.nn as nn

cuda.init() 
print("Current GPU: ", torch.cuda.current_device())
print("Num of Devices: ", cuda.Device.count())

a = torch.cuda.FloatTensor([1, 2]) 
print(a.get_device(), type(a))

sq = nn.Sequential(
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 4),
        nn.Softmax()
)
model = sq.cuda()
print("here", next(model.parameters()).is_cuda)
#print("How much memory is allocated: ", torch.cuda.max_memory_allocated())

'''
cuda0 = torch.cuda.device(0)
cuda1 = torch.cuda.device(1)

x = torch.Tensor([1, 2]).to(cuda0)
y = torch.Tensor([3, 4]).to(cuda1)

print("x: ", x.get_device())
print("y: ", y.get_device())
'''

torch.cuda.set_device(1) #HOW TO CHANGE GPU DEFAULT DEVICE SETTING
print("Current GPU: ", torch.cuda.current_device())

b = torch.cuda.FloatTensor([1, 2]) 
print("here here", b.get_device(), type(b))
sq2 = nn.Sequential(
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 4),
        nn.Softmax()
)
print("before loading onto GPU", type(sq2))
model = sq2.cuda()
print("after loading onto GPU", type(model), model.get_device())
print(next(model.parameters()).is_cuda)


cuda0 = torch.cuda.device('cuda:0')
cuda1 = torch.cuda.device('cuda:1')
cuda2 = torch.cuda.device('cuda:2')

x = torch.cuda(cuda2).Tensor([1., 2.])
