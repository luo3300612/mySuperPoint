# mySuperPoint
数据集生成方法是在[Superpoint](https://github.com/rpautrat/SuperPoint)中Synthetic dataset的生成方法

## tf版实现笔记
* 网络结构与pytorch版一样

训练
```shell
python experiment.py train configs/magic-point_shapes.yaml magic-point_synth
```
第一个参数是指定进行训练，第二个参数是配置文件目录

第一次训练时，会自动生成数据集

experience.py(line 150) - > _cli_train(line 81) 
-> train(line 19) -> _init_graph(line63)
-> get_dataset -> datasets/__init__.py 返回SyntheticShapes(**config['data'])，
SyntheticShape的__init__方法会合并类自带的default_config和config['data']这两个字典作为
最后的config，然后调用_init_dataset和_get_data生成数据集,其中_init_dataset负责划分文件目录，
然后用dump_primitive_data来预处理图片

* bn层竟然放在激活函数的后面，但是网上说放在激活层后面效果也不错？
* tf实现版本有batch normalization

#### TODO
* 浏览数据集，记下笔记 done
* 测试数据集生成方法 done
* 找到原数据集生成的代码 done
* 根据实际场景，结合Opencv函数，找到扩展数据集的方法
* 需要有label生成前后的对比 done
* 加上TensorboardX可视化训练过程
* 正负样本差太多导致网络倾向全部判定为无关键点
* 即便可能关键点的激活值很小，也可以保留topk个作为参考
* pixel shuffle in pytorch
* pretrained model的置信度的门限值竟然是0.015,tensroflow版的置信度门限是0.001,或许门限设低一点可以解决问题？
* 并不行，检测出的图片兴趣点分布在边框上？发现pretrained model有一个去除边框上一定范围内的点的操作，为什么？
* 借鉴tf实现中的使用argmax来做label生成？
* tf版本
* 调查pretrained model的输出scale的范围

## Reference
* [Superpoint](https://github.com/rpautrat/SuperPoint)，基于tensorflow的SuperPoint
* [SuperpointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)，
作者预训练的SuperPoint模型，可以调用到图片、视频和摄像头上，但作者明确说明不会开放训练代码、合成数据集
* [demoasd](https://www.youtube.com/watch?v=gtzxuET74Mk) ,youtube上的视频Demo，在高分辨率的情况下似乎有很大的问题
* [关于损失函数](https://zhuanlan.zhihu.com/p/54969632)