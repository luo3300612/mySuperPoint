# mySuperPoint
## log
### 2019-02-01
在[Superpoint](https://github.com/rpautrat/SuperPoint)中寻找Synthetic dataset的生成方法
#### 生成数据集笔记
生成的数据集有9类，分别是

| 类别 | 每张图的个数 | 备注 |
| ------ |: ------ :| ------ |
|  checkerboard | 1 | |
| cube |1||
| ellipse| 多||
| line| 多||
| multiple polygon |多||
|polygon |1|多为三角形|
|star |1|实际不是星星，是线段交成的结形
|stripes| 多||
|gaussian noise|?|噪声|
其中，每类包含训练集10000，测试集500，验证集200

#### TODO
* 浏览数据集，记下笔记
* 测试数据集生成方法
* 找到原数据集生成的代码
* 根据实际场景，结合Opencv函数，找到扩展数据集的方法

## Reference
* [Superpoint](https://github.com/rpautrat/SuperPoint)，基于tensorflow的SuperPoint
* [SuperpointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)，
作者预训练的SuperPoint模型，可以调用到图片、视频和摄像头上，但作者明确说明不会开放训练代码、合成数据集
* [demoasd](https://www.youtube.com/watch?v=gtzxuET74Mk) ,youtube上的视频Demo，在高分辨率的情况下似乎有很大的问题

## 问题
* 圆心是否算是关键点