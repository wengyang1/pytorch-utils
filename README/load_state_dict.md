## demo
```commandline
在PyTorch中加载自己训练过的ResNet模型权重是一个相对直接的过程。这里，我将通过几个步骤来指导你如何完成这个过程。

1. 定义ResNet模型
首先，你需要定义或加载一个ResNet模型。如果你已经有一个自定义的ResNet模型，可以直接使用它。如果没有，你可以使用PyTorch的torchvision.models模块中的ResNet，然后根据你的需要进行修改。

python
import torchvision.models as models  
  
# 加载预定义的ResNet模型，这里以ResNet18为例  
resnet = models.resnet18(pretrained=False)  # pretrained=False 表示不使用预训练的权重  
  
# 如果你需要修改模型（比如改变输出层的类别数），可以在这里进行  
num_ftrs = resnet.fc.in_features  
resnet.fc = torch.nn.Linear(num_ftrs, your_num_classes)  # 替换全连接层
2. 加载权重
假设你已经将训练好的权重保存到了一个文件中（比如使用torch.save(resnet.state_dict(), 'resnet_weights.pth')）。现在，你可以使用torch.load()来加载这些权重，并使用load_state_dict()方法将权重加载到你的模型中。

python
# 加载权重  
checkpoint = torch.load('resnet_weights.pth')  
  
# 将权重加载到模型中  
resnet.load_state_dict(checkpoint)  
  
# 注意：如果你的模型在训练后进行了修改（比如添加了新的层或删除了某些层），  
# 那么加载权重时可能会遇到键不匹配的问题。在这种情况下，你可能需要  
# 从checkpoint中删除或修改不匹配的键。  
  
# 例如，如果你删除了某个层，并且不想加载该层的权重，你可以这样做：  
# del checkpoint['被删除的层的键']  
# resnet.load_state_dict(checkpoint, strict=False)  # 使用strict=False来忽略不匹配的键
3. 使用模型进行推理
加载权重后，你的模型就已经准备好进行推理了。你可以将输入数据传递给模型，并获取输出。

python
# 假设input_tensor是你的输入数据  
# input_tensor = ...  
  
# 将模型设置为评估模式  
resnet.eval()  
  
# 如果有必要，将输入数据移动到正确的设备（CPU或GPU）  
# input_tensor = input_tensor.to(device)  
  
# 进行推理  
with torch.no_grad():  
    output = resnet(input_tensor)  
  
# 处理输出...
注意事项
确保在加载权重之前，模型的架构与保存权重时使用的架构相匹配。
如果在保存权重后修改了模型架构（比如改变了层数或添加了新的层），则可能需要手动调整加载的权重以匹配新的架构。
使用torch.load()加载权重时，默认情况下，权重会被加载到与保存时相同的设备上（CPU或GPU）。如果需要在不同的设备上使用这些权重，请确保在加载后将其移动到正确的设备上。
如果在加载权重时遇到RuntimeError: Error(s) in loading state_dict for ...错误，这通常意味着模型架构与权重不匹配。检查你的模型定义和保存的权重文件，确保它们是一致的。
```
## 模型不匹配
```commandline
这个错误信息表明在尝试将预训练的权重加载到你的ResNet模型时，权重文件中包含了一些不在你的模型state_dict中的键。这些额外的键（如"total_ops", "total_params", 以及各层后的.total_ops和.total_params）很可能是由某些训练或模型分析工具添加的，用于记录模型的计算量或参数数量，而不是模型本身的权重参数。

为了解决这个问题，你可以在加载权重之前从权重字典中删除这些额外的键。这可以通过遍历权重字典并删除不匹配的键来实现。以下是一个示例代码，展示了如何在加载权重之前处理这个问题：

python
import torch  
  
# 假设resnet是你的模型实例，checkpoint是加载的权重字典  
checkpoint = torch.load('resnet_weights.pth')  
  
# 创建一个新的字典来存储有效的权重  
model_state_dict = {}  
  
# 遍历checkpoint中的每个键  
for key, value in checkpoint.items():  
    # 检查键是否以'module.'开头（这通常发生在模型被nn.DataParallel包装时）  
    # 或者检查键是否不包含'total_ops'和'total_params'  
    if not key.startswith('module.') and not any(unwanted in key for unwanted in ['total_ops', 'total_params']):  
        # 如果键是有效的，则将其添加到model_state_dict中  
        model_state_dict[key] = value  
  
# 现在，使用清理后的字典来加载权重  
resnet.load_state_dict(model_state_dict, strict=False)  # 使用strict=False来忽略不匹配的键（尽管在这个例子中我们已经手动清理了它们）  
  
# 注意：如果你确定你的模型没有被nn.DataParallel包装，那么你可以省略对'module.'的检查
然而，在这个特定的例子中，由于你已经手动清理了所有不相关的键，所以strict=False可能不是必需的。但是，保留它作为一个选项总是一个好主意，以防万一你的模型架构与权重文件之间还有其他微小的不匹配。

另外，请注意，如果你的模型在训练时使用了nn.DataParallel进行并行处理，那么权重文件中的键可能会以'module.'开头。在这种情况下，你可能需要在将键添加到model_state_dict之前，从每个键中删除'module.'前缀（如果你的当前模型没有使用nn.DataParallel）。这可以通过简单的字符串替换来实现：

python
if key.startswith('module.'):  
    key = key[7:]  # 移除'module.'前缀
但是，请注意，如果你的模型当前没有使用nn.DataParallel，并且权重文件中的键包含'module.'前缀，那么这可能意味着权重是为并行模型训练的，而你现在正在尝试将其加载到非并行模型中。在这种情况下，你需要确保你的模型架构与权重文件完全匹配，包括是否使用并行处理。
```