# 05 · 语音与音频生成架构全解

> 目标：理解 TTS、ASR、音乐生成背后的模型架构——
> 从传统语音合成到 Codec Language Model（VALL-E）再到 Flow Matching TTS（F5-TTS）。
> 2024–2026 年的语音 AI 正在经历从“离线生成”到“实时多模态交互”的范式转变。

---

## 1. 音频 / 语音在 AIGC 中的位置

```
AIGC 音频领域地图：

┌──────────────────────────────────────────┐
│              AIGC 音频                    │
├────────────┬───────────┬─────────────────┤
│   语音合成  │  语音识别  │   音乐/音效生成  │
│   (TTS)    │  (ASR)    │                 │
├────────────┼───────────┼─────────────────┤
│ CosyVoice  │ Whisper   │ MusicGen        │
│ F5-TTS     │           │ Stable Audio    │
│ ChatTTS    │           │ Udio / Suno     │
│ Fish Speech│           │                 │
│ VALL-E     │           │                 │
└────────────┴───────────┴─────────────────┘
```

### 1.1 音频任务的四条链路

| 链路 | 输入输出 | 代表应用 | 架构重点 |
|---|---|---|---|
| ASR | speech → text | 会议转录、字幕、语音搜索 | 鲁棒性、时间戳、说话人、长音频分段 |
| TTS | text + voice prompt → speech | 配音、有声书、客服 | 音色、韵律、可懂度、流式延迟 |
| Audio LM | audio/text → audio/text | 音频问答、声音事件理解 | 音频编码器、LLM 对齐、多任务数据 |
| Music/SFX | text/audio → music/effect | 音乐创作、游戏音效 | 长时结构、版权、风格控制 |

实时语音助手通常是 ASR + LLM + TTS + interruption/barge-in + VAD 的系统工程，不只是一个 TTS 模型。

---

## 2. 音频基础知识

### 2.1 关键概念

```
波形 (Waveform):
  音频信号的原始表示
  x[n] = 在第 n 个采样点的幅值
  维度: (num_samples,)

采样率 (Sample Rate):
  每秒采样次数
  - 电话: 8,000 Hz
  - 语音常用: 16,000 Hz
  - CD 音质: 44,100 Hz
  - 高保真: 48,000 Hz

  1 秒 16kHz 音频 = 16,000 个采样点

频谱图 (Spectrogram):
  时间-频率的二维表示
  用 STFT (Short-Time Fourier Transform) 计算
  维度: (freq_bins, time_frames)

梅尔频谱图 (Mel-Spectrogram):
  频率轴换成梅尔刻度（模拟人耳感知）
  维度: (n_mels, time_frames)，常见 n_mels=80
```

```python
import torchaudio
import torchaudio.transforms as T

# 加载音频
waveform, sample_rate = torchaudio.load("speech.wav")
print(f"采样率: {sample_rate}, 形状: {waveform.shape}")
# 例如: 采样率: 16000, 形状: torch.Size([1, 48000]) → 3 秒

# 计算梅尔频谱图
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)
mel_spec = mel_transform(waveform)
print(f"梅尔频谱: {mel_spec.shape}")
# 例如: torch.Size([1, 80, 188]) → 80 个梅尔频带, 188 个时间帧

# 转为 log scale（标准做法）
log_mel = torch.log(mel_spec + 1e-8)
```

### 2.2 音频表示的层次

```
从"原始"到"抽象"：

原始波形 (16kHz, 1D)
  │  每秒 16,000 个采样点
  ▼
梅尔频谱图 (2D)
  │  每秒 ~62 帧 (hop=256, sr=16000)
  ▼
神经音频编解码 (离散 token)
  │  每秒 ~50–75 个 token（多层 codebook）
  ▼
语义 token
  │  每秒 ~25–50 个 token
  ▼
文本 token
     极少量 token
```

### 2.3 音频表示决定延迟

| 表示 | 适合 | 延迟特点 | 主要风险 |
|---|---|---|---|
| waveform | 端到端音频模型、vocoder | 数据量最大，实时难度高 | 计算贵、长程依赖难 |
| mel spectrogram | TTS、ASR | 生态成熟，可分块处理 | 需要 vocoder，音质受限 |
| codec token | TTS、音乐、语音 LM | token 化后可用 LM/streaming | codec 失真、codebook 层间依赖 |
| semantic token | 语义/内容建模 | token 少，适合高层规划 | 细节和音色需另一路补充 |

做实时系统时要看 **首包延迟**、**实时率 RTF** 和 **流式稳定性**，不能只看离线 MOS。

---

## 3. TTS 技术演进

### 3.1 四代 TTS

```
第一代: 拼接合成 (Concatenative)
  把录好的音素片段拼接起来
  例如: "你好" = "n" + "i" + "h" + "ao"
  效果: 机械、不自然

第二代: 参数合成 (Parametric)
  用统计模型预测声学参数
  例如: HMM-TTS
  效果: 流畅但音质差

第三代: 神经网络 TTS
  端到端学习: 文本 → 梅尔频谱 → 波形
  例如: Tacotron 2 + WaveNet / HiFi-GAN
  效果: 接近真人，但需要大量单人数据

第四代: Zero-shot TTS (2023–现在)
  只需 3–10 秒参考音频，就能克隆任意说话人的声音
  例如: VALL-E, CosyVoice, F5-TTS
  效果: 高度自然，且能模仿未见过的说话人
```

### 3.2 传统神经 TTS 架构

```
Tacotron 2 + HiFi-GAN (2018–2022 主流):

  文本 → Tacotron 2 (Seq2Seq) → 梅尔频谱 → HiFi-GAN (声码器) → 波形

  Tacotron 2: Encoder (文本) → Attention → Decoder (梅尔频谱)
  HiFi-GAN: 将梅尔频谱转换为波形（GAN-based vocoder）

  局限:
  - 需要几十小时单人录音
  - 只能合成训练过的声音
  - 自回归解码速度慢
```

---

## 4. 神经音频编解码器：TTS 的范式转变

### 4.1 核心思想

**把连续音频波形编码为离散 token——让 LLM 范式（next-token prediction）直接用于音频生成。**

```
传统 TTS:
  文本 → 梅尔频谱 (连续) → 波形 (连续)

Codec-based TTS:
  文本 → 离散音频 token → 解码器 → 波形

关键insight:
  如果音频可以用离散 token 表示，那么语音合成就变成了"语言模型"问题！
  文本 token → 音频 token，和 GPT 生成文本完全一样。
```

### 4.2 EnCodec (Meta, 2022)

EnCodec 是最有影响力的神经音频编解码器：

```
EnCodec 架构：

  波形 → Encoder (1D Conv) → 连续 latent → RVQ → 离散 token → Decoder → 波形

  RVQ (Residual Vector Quantization, 残差向量量化):
    第 1 层 codebook: 量化 latent → 粗粒度 token (大致轮廓)
    第 2 层 codebook: 量化残差 → 更细节
    第 3 层 codebook: 量化残差 → 更细节
    ...
    第 8 层 codebook: 最精细的细节

  每层 codebook 大小: 1024 (10 bits)
  码率: 1.5 kbps → 6 kbps → 12 kbps → 24 kbps
```

```python
# EnCodec 使用示例
from transformers import EncodecModel, AutoProcessor

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# 编码: 波形 → 离散 token
inputs = processor(raw_speech=waveform, sampling_rate=24000, return_tensors="pt")
encoder_outputs = model.encode(inputs["input_values"])
audio_codes = encoder_outputs.audio_codes
print(audio_codes.shape)  # (batch, n_codebooks, n_frames)
# 例如: (1, 8, 150) → 8 层 codebook, 150 个时间步

# 解码: 离散 token → 波形
decoded = model.decode(audio_codes, encoder_outputs.audio_scales)
reconstructed_waveform = decoded.audio_values
```

### 4.3 DAC (Descript Audio Codec)

DAC 是 EnCodec 的改进版，在高保真度方面表现更好：

| 特性 | EnCodec | DAC |
|---|---|---|
| 采样率 | 24 kHz | 44.1 kHz |
| Codebook 层数 | 8 | 9 |
| 压缩率 | 320x | 512x |
| 音质 | 好 | 更好（尤其音乐） |

---

## 5. VALL-E：Codec Language Model for TTS

### 5.1 核心思想

VALL-E (Wang et al., 2023) 的突破性idea：
**把 TTS 当作语言模型——给定文本 + 3 秒参考音频 → 生成目标音频的 codec token。**

```
VALL-E 架构：

输入:
  - 文本 token: "Hello, how are you?"
  - 参考音频 codec token: [3 秒参考音频的 EnCodec token]

输出:
  - 目标音频的 codec token → EnCodec 解码 → 波形

模型结构:
  第 1 层 codebook token: 自回归 Transformer (AR)
    → 逐 token 生成，控制韵律和内容

  第 2–8 层 codebook token: 非自回归 Transformer (NAR)
    → 一次性生成，补充音频细节

  ┌─────────────────────┐
  │  AR Model (Layer 1)  │ ← 逐 token 生成
  │  phonemes + prompt   │
  └──────────┬──────────┘
             │ Layer 1 codes
             ▼
  ┌─────────────────────┐
  │  NAR Model (L2–L8)  │ ← 并行生成
  │  条件: Layer 1 codes │
  └──────────┬──────────┘
             │ All layer codes
             ▼
  ┌─────────────────────┐
  │   EnCodec Decoder    │ → 波形
  └─────────────────────┘
```

### 5.2 VALL-E 的意义

| 能力 | 传统 TTS | VALL-E |
|---|---|---|
| 需要训练数据 | 几十小时单人录音 | 3 秒参考音频 |
| 说话人范围 | 训练过的声音 | **任意说话人** (zero-shot) |
| 情感/韵律 | 有限控制 | 从参考音频中保留 |
| 训练数据规模 | 几十小时 | 60,000 小时（LibriLight） |

---

## 6. CosyVoice

CosyVoice (FunAudioLLM, 阿里, 2024) 是高质量的开源零样本 TTS。

### 6.1 架构

```
CosyVoice 架构：

  ┌──────────────────────────────────────────┐
  │  Text Encoder (LLM-based)                │
  │  文本 → 语义 token                       │
  └────────────────┬─────────────────────────┘
                   │
  ┌────────────────┴─────────────────────────┐
  │  Flow Matching Decoder                    │
  │  语义 token + 说话人 embedding             │
  │  → 梅尔频谱                               │
  └────────────────┬─────────────────────────┘
                   │
  ┌────────────────┴─────────────────────────┐
  │  HiFi-GAN Vocoder                        │
  │  梅尔频谱 → 波形                          │
  └──────────────────────────────────────────┘
```

### 6.2 关键特性

| 特性 | 说明 |
|---|---|
| **零样本语音克隆** | 3–10 秒参考音频即可 |
| **跨语言合成** | 中文说话人说英文（保持音色） |
| **指令控制** | 支持情感、语速等自然语言指令 |
| **流式推理** | 支持实时流式生成 |
| **开源** | 模型和代码完全开源 |

### 6.3 CosyVoice 2 的方向

CosyVoice 2 进一步强调流式语音生成和大规模语音建模。
它说明 TTS 的竞争点正在从“能不能克隆声音”转向：

- 低首包延迟；
- 长文本韵律稳定；
- 跨语言和方言；
- 情绪/语气可控；
- 和 LLM 对话系统的端到端衔接；
- 安全水印和说话人授权。

```python
# CosyVoice 使用示例
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice("pretrained_models/CosyVoice-300M")

# 零样本语音克隆
prompt_speech = load_wav("reference_3sec.wav", 16000)
output = cosyvoice.inference_zero_shot(
    tts_text="你好，欢迎使用 CosyVoice 语音合成系统。",
    prompt_text="这是参考音频的文字内容。",
    prompt_speech_16k=prompt_speech,
)

for i, result in enumerate(output):
    torchaudio.save(f"output_{i}.wav", result["tts_speech"], 22050)

# 指令控制
output = cosyvoice.inference_instruct(
    tts_text="今天天气真好啊！",
    spk_id="中文女",
    instruct_text="用开心的语气说",
)
```

---

## 7. F5-TTS

F5-TTS (Chen et al., 2024) 使用 Flow Matching 的非自回归 TTS。

### 7.1 核心创新

```
F5-TTS 的核心思路：

  不用自回归！整个语音一次性生成。

  输入: 文本 + 参考音频（拼接在一起）
  过程: Flow Matching（从噪声直接生成梅尔频谱）
  输出: 完整的梅尔频谱 → 声码器 → 波形

  ┌────────────────────────────────────────┐
  │   [参考音频梅尔频谱] [文本对应的噪声]    │
  │              ↓                         │
  │     DiT (Flow Matching Transformer)    │
  │              ↓                         │
  │   [参考音频梅尔频谱] [生成的梅尔频谱]    │
  └────────────────────────────────────────┘
```

| 特性 | 说明 |
|---|---|
| **非自回归** | 一次性生成所有帧，速度快 |
| **Flow Matching** | 训练目标简单，质量高 |
| **In-context learning** | 参考音频和目标拼在一起，模型自动学习说话人特征 |
| **E2 TTS 简化** | 去掉了复杂的时间对齐模块 |

---

## 8. ChatTTS

ChatTTS (2024) 专为对话场景设计的 TTS 模型。

### 8.1 特点

| 特性 | 说明 |
|---|---|
| **对话式韵律** | 自然的对话语气，不像朗读 |
| **笑声/停顿** | 支持插入笑声 `[laugh]`、停顿 `[break]` 等 |
| **多说话人** | 通过 speaker embedding 控制 |
| **流式输出** | 支持实时生成 |
| **中英文** | 原生支持中英文 |

```python
import ChatTTS
import torch

chat = ChatTTS.Chat()
chat.load(compile=False)

# 基本使用
texts = ["你好啊，今天天气真不错！[laugh] 我们出去走走吧。"]

wavs = chat.infer(
    texts,
    params_infer_code=ChatTTS.Chat.InferCodeParams(
        temperature=0.3,
        top_P=0.7,
    ),
)

# 保存
import torchaudio
torchaudio.save("output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
```

---

## 9. Fish Speech / MaskGCT

### 9.1 Fish Speech

Fish Speech (2024) 是另一个优秀的开源 TTS：

| 特性 | 说明 |
|---|---|
| 方法 | VITS2-based + Large-scale Training |
| 零样本 | 支持，15 秒参考 |
| 语言 | 中英日韩等多语言 |
| 速度 | 实时率 RTF < 0.1（非常快） |
| 特色 | 极低延迟，适合实时应用 |

### 9.2 MaskGCT

MaskGCT (Masked Generative Codec Transformer, 2024) 是一种**非自回归**的 codec TTS：

```
MaskGCT 的思路：

  不是逐 token 生成，而是用 Masked 预测（类似 BERT 的思路）：
  1. 先生成语义 token（粗粒度）
  2. 用 mask-and-predict 策略逐步生成 codec token
  3. 每次预测最有信心的 token，mask 掉不确定的

  好处：
  - 比自回归快（可以并行）
  - 比一次性生成质量好（渐进式精化）
```

---

## 10. Whisper：ASR 的事实标准

### 10.1 架构

Whisper (Radford et al., 2022) 是 OpenAI 的语音识别模型：

```
Whisper 架构：

  音频波形 (30 秒)
       │
       ▼
  梅尔频谱 (80×3000)
       │
       ▼
  ┌──────────────────┐
  │  Audio Encoder    │  Conv1D + Transformer Encoder
  │  (编码器)         │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Text Decoder     │  Transformer Decoder (自回归)
  │  (解码器)         │  交叉注意力连接 encoder
  └────────┬─────────┘
           │
           ▼
  文本输出: "Hello, how are you?"
```

### 10.2 关键特性

| 特性 | 说明 |
|---|---|
| 架构 | Encoder-Decoder Transformer |
| 训练数据 | 680,000 小时多语言音频 |
| 多任务 | 语音识别 + 翻译 + 语言检测 + 时间戳 |
| 模型规格 | tiny (39M) → large-v3 (1.5B) |
| 多语言 | 99 种语言 |

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device_map="auto",
)

result = pipe("audio.wav", return_timestamps=True)
print(result["text"])
# "Hello, how are you today?"
print(result["chunks"])
# [{"timestamp": (0.0, 1.5), "text": "Hello, how are you today?"}]
```

### 10.3 Whisper 模型规格对比

| 模型 | 参数 | 英文 WER | 相对速度 | 推荐场景 |
|---|---|---|---|---|
| tiny | 39M | ~7.6% | 32x | 实时/边缘设备 |
| base | 74M | ~5.0% | 16x | 低资源场景 |
| small | 244M | ~3.4% | 6x | 平衡选择 |
| medium | 769M | ~2.9% | 2x | 高精度 |
| large-v3 | 1.5B | ~2.0% | 1x | 最高精度 |

---

## 11. 语音克隆

### 11.1 Zero-shot vs Few-shot

| 方式 | 数据需求 | 质量 | 代表 |
|---|---|---|---|
| **Zero-shot** | 3–10 秒参考 | 良好 | VALL-E, CosyVoice |
| **Few-shot** | 1–5 分钟 | 更好 | 微调方法 |
| **Fine-tuned** | 30 分钟+ | 最佳 | 传统 TTS 微调 |

### 11.2 语音克隆的伦理问题

```
⚠️ 重要伦理警告：

语音克隆技术可以被滥用于：
  - 电话诈骗（模仿亲友声音）
  - Deepfake 音频
  - 未经授权的声音复制

负责任的使用：
  - 仅在获得声音所有者授权时使用
  - 添加水印标记 AI 生成的语音
  - 遵守当地法律法规
```

---

## 12. 音乐生成

### 12.1 MusicGen (Meta, 2023)

MusicGen 的架构与 VALL-E 类似——基于 EnCodec 的 codec language model：

```
MusicGen 架构：

  文本描述: "upbeat electronic dance music with heavy bass"
       │
       ▼
  ┌──────────────────┐
  │  T5 Text Encoder  │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Transformer LM   │  自回归生成 EnCodec token
  │  (条件: 文本)     │  "delayed pattern" 策略
  └────────┬─────────┘
           │ codec tokens
           ▼
  ┌──────────────────┐
  │  EnCodec Decoder  │  → 波形
  └──────────────────┘
```

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["happy rock song with electric guitar"],
    padding=True,
    return_tensors="pt",
)

# 生成 10 秒音乐
audio_values = model.generate(**inputs, max_new_tokens=512)  # 512 tokens ≈ 10 秒

# 保存
import scipy.io.wavfile
scipy.io.wavfile.write(
    "music.wav",
    rate=model.config.audio_encoder.sampling_rate,
    data=audio_values[0, 0].numpy(),
)
```

### 12.2 Stable Audio (Stability AI, 2024)

Stable Audio 使用 **Latent Diffusion** 生成音乐：

| 特性 | MusicGen | Stable Audio |
|---|---|---|
| 方法 | Codec LM (自回归) | Latent Diffusion |
| 时长 | ~30 秒 | ~95 秒 (v2) |
| 控制 | 文本描述 | 文本 + 时长 + 风格 |
| 质量 | 好 | 更好（44.1 kHz） |

### 12.3 Udio / Suno

Udio 和 Suno 是 2024 年最受关注的商业音乐生成产品：

| 特性 | Udio | Suno |
|---|---|---|
| 生成方式 | 文本描述 | 文本描述 + 歌词 |
| 音乐质量 | 接近专业制作 | 接近专业制作 |
| 歌曲结构 | 完整（前奏/副歌/结尾） | 完整 |
| 人声 | 支持 | 支持 |
| 开源 | ❌ | ❌ |

---

## 13. 音频理解：Audio-Language Models

### 13.1 趋势

音频理解正在向多模态大模型整合：

```
独立模型时代:
  ASR: 音频 → 文本（只做转录）
  分类: 音频 → 标签（只做分类）

Audio-Language Model 时代:
  音频 + 文本 → 文本（理解+推理+问答）

  "这段音频里在播放什么乐器？"
  "这个人的情绪听起来怎样？"
  "总结这段会议录音的要点。"
```

### 13.2 代表模型

| 模型 | 机构 | 能力 |
|---|---|---|
| Qwen-Audio | 阿里 | 语音理解 + 音频理解 + 多轮对话 |
| Qwen2-Audio | 阿里 | 升级版，支持更多音频任务 |
| SALMONN | 字节 | 语音 + 音频 + 音乐理解 |
| Gemini | Google | 原生音频理解（端到端） |
| GPT-4o | OpenAI | 端到端语音对话 |
| Qwen2.5-Omni | 阿里 | 多模态输入，支持流式语音输出 |

```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": "audio.wav"},
            {"type": "text", "text": "这段音频里说了什么？"},
        ],
    },
]

text = processor.apply_chat_template(conversation, tokenize=False)
inputs = processor(text=text, audios=[audio], return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=256)
```

### 13.3 端到端语音模型 vs 级联系统

| 方案 | 优点 | 缺点 |
|---|---|---|
| ASR → LLM → TTS | 可解释、可替换、易审计、文本日志完整 | 延迟高，语气和中断处理不自然 |
| 端到端 audio-language | 低延迟、保留语气/情绪、交互自然 | 调试难、可控性和审计更复杂 |
| 混合方案 | 关键路径端到端，旁路输出文本和日志 | 系统复杂，但更接近产品需求 |

企业落地通常先用级联系统，因为可观测性和合规更好；实时陪伴、同传、语音 Agent 则更需要端到端或混合架构。

---

## 14. 评估指标

### 14.1 TTS 评估

| 指标 | 全称 | 衡量什么 | 方向 |
|---|---|---|---|
| **MOS** | Mean Opinion Score | 人类主观打分 (1–5) | 越高越好 |
| **WER** | Word Error Rate | 合成语音的可懂度 | 越低越好 |
| **Speaker Similarity** | 说话人相似度 | 克隆声音是否像 | 越高越好 |
| **PESQ** | Perceptual Evaluation of Speech Quality | 客观音质 | 越高越好 |
| **UTMOS** | 自动 MOS 预测 | 用模型预测 MOS | 越高越好 |
| **RTF** | Real-Time Factor | 生成耗时 / 音频时长 | 越低越好 |
| **首包延迟** | Time to first audio chunk | 用户多久听到第一段声音 | 越低越好 |

```
MOS (Mean Opinion Score) 评估流程：
  1. 准备 20+ 条合成语音
  2. 20+ 位评估者（母语者）
  3. 每人对每条打分: 1(差) → 5(优秀)
  4. 取平均值

  MOS 参考值:
    真人录音: ~4.5
    优秀 TTS: ~4.0–4.3
    一般 TTS: ~3.5–4.0
    差的 TTS: < 3.5
```

### 14.2 ASR 评估

```python
# WER (Word Error Rate) 计算
# WER = (S + D + I) / N
# S: 替换错误, D: 删除错误, I: 插入错误, N: 参考文本词数

import jiwer

reference = "hello how are you doing today"
hypothesis = "hello how are you doing"

error = jiwer.wer(reference, hypothesis)
print(f"WER: {error:.2%}")  # WER: 16.67% (1 deletion / 6 words)
```

### 14.3 语音系统评估不要只看单指标

| 场景 | 主指标 | 还要看 |
|---|---|---|
| 客服 ASR | WER / CER | 噪声、口音、专有名词、时间戳 |
| 会议纪要 | DER + WER | 说话人分离、长音频漂移、摘要准确性 |
| 配音 TTS | MOS | 情绪、停顿、长文本一致性、版权授权 |
| 实时助手 | 首包延迟 + RTF | 打断响应、流式稳定、端到端延迟 |
| 语音克隆 | speaker similarity | 内容可懂度、泄露风险、水印 |
| 音乐生成 | 主观偏好 | 长时结构、版权相似性、人声质量 |

---

## 15. 实际应用场景

| 应用 | 技术组合 | 说明 |
|---|---|---|
| **虚拟助手** | ASR + LLM + TTS | Siri、小爱同学的核心链路 |
| **有声书** | 长文本 TTS | 需要韵律自然、不疲劳 |
| **视频配音** | TTS + 跨语言 | 电影/短视频多语言配音 |
| **直播数字人** | ASR + LLM + TTS + 驱动 | 7×24 直播 |
| **会议纪要** | ASR + LLM | 实时转录 + 摘要 |
| **音乐创作辅助** | Music Gen | 快速原型、背景音乐 |

### 实际 pipeline 示例

```python
# 完整的语音对话 pipeline（概念代码）
def voice_chat(audio_input):
    # 1. ASR: 语音 → 文本
    text = whisper_pipeline(audio_input)["text"]

    # 2. LLM: 文本 → 回复文本
    response = llm.generate(text)

    # 3. TTS: 文本 → 语音
    audio_output = cosyvoice.infer(response)

    return audio_output
```

---

## 16. TTS 技术对比

| 模型 | 方法 | 零样本 | 开源 | 中文 | 英文 | 实时 | 音质 |
|---|---|---|---|---|---|---|---|
| VALL-E | Codec LM (AR+NAR) | ✅ | ❌ | ❌ | ✅ | ❌ | ★★★★ |
| CosyVoice | LLM + Flow Matching | ✅ | ✅ | ✅ | ✅ | ✅ | ★★★★★ |
| F5-TTS | Flow Matching (NAR) | ✅ | ✅ | ✅ | ✅ | ✅ | ★★★★ |
| ChatTTS | Codec LM | ✅ | ✅ | ✅ | ✅ | ✅ | ★★★★ |
| Fish Speech | VITS2-based | ✅ | ✅ | ✅ | ✅ | ✅ | ★★★★ |
| MaskGCT | Masked Codec LM | ✅ | ✅ | ✅ | ✅ | ✅ | ★★★★ |

---

## 17. 常见坑

### 17.1 采样率不匹配

```python
# ❌ 常见错误：模型需要 16kHz，但输入是 44.1kHz
waveform, sr = torchaudio.load("audio.wav")  # sr=44100
model_output = model(waveform)  # 预期 16kHz → 完全错误的结果

# ✅ 正确做法：重采样
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)
```

### 17.2 音频长度不对

不同模型对输入长度有限制：
- Whisper：30 秒（长音频需要分段）
- CosyVoice 参考音频：3–10 秒
- ChatTTS：单次不超过 ~30 秒

### 17.3 以为 TTS 只需要关注音质

好的 TTS 不只是"听起来清楚"：
- 韵律（节奏、重音、语调）
- 情感表达
- 停顿和呼吸感
- 多说话人一致性

### 17.4 忽视音频编解码质量

Codec-based TTS 的音质上限取决于 codec 的重建质量。
如果 EnCodec/DAC 重建就有瑕疵（金属感、模糊），TTS 模型再好也救不回来。

### 17.5 中文 TTS 的多音字和韵律

中文 TTS 比英文更难：
- 多音字："行"→ háng / xíng
- 韵律模式不同于英文
- 需要专门的中文前端（text normalization + G2P）

### 17.6 混淆 RVQ 的不同层

EnCodec 的 8 层 codebook 不是平等的：
- 第 1 层：最重要，包含语义和韵律信息
- 后面的层：补充声学细节（音色、清晰度）
VALL-E 的 AR 模型只生成第 1 层，正是因为它最关键。

### 17.7 忽视授权、水印和滥用风险

语音克隆比图像生成更敏感，因为声音直接绑定个人身份。
上线前至少明确：

- 是否获得说话人授权；
- 是否限制相似度过高的克隆；
- 是否添加可检测水印；
- 是否保留生成记录；
- 是否阻断诈骗、冒充、政治/金融敏感场景；
- 是否支持用户撤销授权和删除声音样本。

这不是“产品条款”附属问题，而是语音模型能否安全使用的一部分。

---

## 18. 实践任务：设计一个实时语音助手链路

写出一页设计：

```text
场景：
ASR 模型：
LLM：
TTS 模型：
VAD：
是否支持打断：
目标首包延迟：
目标 RTF：
日志和审计：
隐私策略：
失败兜底：
```

再对比两种方案：

| 方案 | 测试点 |
|---|---|
| ASR → LLM → TTS 级联 | 可观测性、延迟、错误传播 |
| 端到端/Omni API | 自然度、打断、成本、可控性 |

建议用同一组 20 条真实语音输入测试，覆盖口音、噪声、打断、长句、专有名词和情绪表达。

---

## 小结

| 概念 | 一句话解释 |
|---|---|
| EnCodec / DAC | 把音频波形编码为离散 token，架起了"LLM for Audio"的桥梁 |
| RVQ | 残差向量量化，多层 codebook 从粗到细编码音频 |
| VALL-E | 把 TTS 变成 codec token 的语言模型问题 |
| CosyVoice | 阿里开源的零样本 TTS，LLM + Flow Matching |
| F5-TTS | 非自回归 Flow Matching TTS，速度快 |
| Whisper | OpenAI 的 ASR 模型，680K 小时数据训练 |
| MusicGen | Meta 的 codec language model for music |
| MOS | TTS 的核心评估指标，人类主观打分 1–5 |

至此，模块 09 的五大主题全部覆盖。建议回到 [README](./README.md) 做自检清单，
确认自己对每个领域的核心概念都能讲清楚。
