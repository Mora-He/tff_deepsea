# tff_deepsea
参考https://github.com/jiawei6636/Bioinfor-DeepSEA 的kera deepsea实现个性化联邦学习deepsea训练。
## model.py
model.py 是deepsea网络结构
model_{$任务名}.py 是各任务的个性化模型（输出层与deepsea结构不同）

## /data
由train、test、valid三种数据构成。通过preprocess.py转为tfrecord格式，为模型学习工作提供所需数据。
数据文件已上传到网盘，下载后放入/data文件夹即可
链接：https://pan.baidu.com/s/19LSBOrui3aEexUsMt35olA 
提取码：tff1 

## loader.py
从train、test、valid三种数据对应数据文件读取数据的工具函数。

## fedPer-deepsea——0510.py
联邦学习deepsea模型

## main.py
通过命令行调用单机训练过程，或模型评估过程。
训练：
`python main.py -e train`
测试：
`python main.py -e test`

main_task.py 是为了进行个性化训练和测试，调用操作同理。
