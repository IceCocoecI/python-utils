
参考
https://milvus.io/docs/zh/image_similarity_search.md

1、
定义特征提取器
然后，我们需要定义一个特征提取器，使用 timm 的 ResNet-34 模型从图像中提取嵌入信息。

执行image_embedding 将训练图片用
cd workspace/
$ bash standalone_embed.sh start