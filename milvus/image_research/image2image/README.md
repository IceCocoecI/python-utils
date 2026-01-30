


安装：https://milvus.io/docs/zh/install_standalone-docker.md
项目：https://milvus.io/docs/zh/image_similarity_search.md


使用步骤
1、启动服务
cd workspace/leanrning/milvus/docker_scripts
bash standalone_embed.sh start

2、先执行image_embedding.py 生成图片特征向量

3、执行image_search.py 搜索图片
     

注意
1、图片后缀为.JPEG，有其他后缀需要修改代码

说明
定义特征提取器
然后，我们需要定义一个特征提取器，使用 timm 的 ResNet-34 模型从图像中提取嵌入信息。