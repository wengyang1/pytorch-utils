# pytorch如何查看模型是否在gpu上
```commandline
在PyTorch中，查看模型是否在GPU上可以通过几种不同的方法来实现。以下是几种常用的方法：

1. 使用model.device属性
如果模型已经被显式地移动到了GPU上（通过model.to('cuda')或model.to(device)，其中device是torch.device('cuda')或类似的设备对象），那么可以通过检查模型的某个参数或缓冲区的.device属性来确定模型当前所在的设备。例如：

python
device = str(model.parameters().next().device)  
print(f"Model is running on {device}.")
或者更简洁地：

python
print(f"Model is running on {next(model.parameters()).device}.")
如果输出是cuda:0（或其他GPU编号），则表示模型在GPU上；如果是cpu，则表示模型在CPU上。

2. 使用.is_cuda属性（已弃用）
需要注意的是，.is_cuda属性在较新版本的PyTorch中已经被弃用，因为它只能判断对象是否在任意GPU上，而不能指定具体的GPU编号。然而，在旧版本的PyTorch中，你可以这样使用：

python
if model.is_cuda:  
    print("模型在 GPU 上运行")  
else:  
    print("模型在 CPU 上运行")
但鉴于其已被弃用，建议优先使用.device属性。

3. 检查模型是否已调用.to('cuda')或.to(device)
在代码中，你可以通过检查是否调用了model.to('cuda')或model.to(device)（其中device是GPU设备）来确定模型是否被移动到了GPU上。这种方法依赖于你的代码逻辑和流程控制。

4. 使用torch.cuda.is_available()检查GPU是否可用
虽然torch.cuda.is_available()函数用于检查GPU是否可用，但它并不直接告诉你模型是否在GPU上。然而，在将模型移动到GPU之前，通常会先检查GPU是否可用。例如：

python
if torch.cuda.is_available():  
    model = model.to('cuda')  
    print("模型已移动到 GPU 上")  
else:  
    print("GPU 不可用，模型将在 CPU 上运行")
总结
为了查看PyTorch中的模型是否在GPU上，推荐使用.device属性来获取模型当前所在的设备信息。.is_cuda属性虽然可用，但因其已被弃用，故不建议在新代码中使用。同时，确保在将模型移动到GPU之前，已经通过torch.cuda.is_available()检查了GPU的可用性。
```