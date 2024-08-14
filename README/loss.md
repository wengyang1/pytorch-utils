## loss可视化
```commandline
想要记录loss等训练指标，可以使用tensorboard
浏览器打开loss日志: tensorboard --logdir=outputs
```
## loss收敛问题
```commandline
1 loss无法收敛，可能的原因
lr大了，之前遇到过一个case，lr=0.001，一直无法收敛，loss反复横跳，改成lr=0.0001很快就收敛了
```
## 二分类
```commandline
torch.nn.BCEWithLogitsLoss
可以参考BCEWithLogitsLoss_demo.py
```
## 多分类
```commandline
torch.nn.CrossEntropyLoss
用法参考 CrossEntropyLoss_demo.py
```