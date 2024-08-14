# backbone优劣
## 运算量 参数量
```commandline
backbone的运算量 参数量有区别
例如resnet18比vgg16更加轻量级,所以训练更快
```
## 残差结构
```commandline
resnet18>vgg16
```
## 数据归一化处理
```commandline
batchNorm
有的backbone对于数据是做了归一化处理的，利于训练的收敛
```
## lr
```commandline
不同的backbone对于lr的敏感程度不同，所以需要选取的lr区间也不同，
resnet18可以选用1e-3,而vgg16则是1e-4(实验总结)
学习率过大会导致模型无法收敛，所以当模型长时间不收敛时，可以考虑降低学习率
```