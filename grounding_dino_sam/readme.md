

# 1) 创建并激活环境
conda create -n grounded-sam python=3.10 -y
conda activate grounded-sam

# 2) 升级 pip
pip install --upgrade pip

# 3) 安装 PyTorch (CUDA 12.4 轮子，适配你 12.6 驱动)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4) 基础依赖
pip install opencv-python numpy supervision
pip install cython pycocotools

# 5） 安装 GroundingDINO，禁用隔离构建
pip install --no-build-isolation "git+https://github.com/IDEA-Research/GroundingDINO.git"

安装 Segment Anything
pip install --no-build-isolation "git+https://github.com/facebookresearch/segment-anything.git"

以上方式报错

新新

# 进入 GroundingDINO 源码根目录
cd /home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grounding_dino_sam/GroundingDINO

# 建议先卸载旧的
pip uninstall -y groundingdino

# 确保环境变量指向你的 CUDA（已有 nvcc 12.6）
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1

# 非编辑安装（会把 _C 放到 site-packages/groundingdino/ 下）
pip install . --no-build-isolation -v


模型下载
cd /home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grounding_dino_sam
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
wget https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


# 如果漏检（目标没检测到）→ 降低阈值
box_threshold=0.20, text_threshold=0.15, min_score=0.25

# 如果误检太多（检测到无关物体）→ 提高阈值  
box_threshold=0.40, text_threshold=0.30, min_score=0.45

# 如果同一目标被重复检测 → 降低 NMS 阈值
nms_threshold=0.3