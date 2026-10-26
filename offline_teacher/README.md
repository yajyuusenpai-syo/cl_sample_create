# Spatial-CLAP spatial teacher 离线运行包

从 GitHub clone 后，先在联网机器的项目根目录运行 `python download_wheelhouse.py`，
按 `pkgs/manifest.json` 恢复被Git忽略的大型wheel文件，再复制项目到离线机。
精简权重及合成 smoke fixture 已包含在仓库，不含真实数据集音频。

目标：Linux x86_64、CPython 3.10、glibc >= 2.28、NVIDIA CUDA 13.0 可用的驱动环境。
固定 PyTorch 2.10.0+cu130。独立 spatial encoder，1,095,616 参数，256 维输出。

## 复制到离线机

只需要复制下面两个目录，保留它们的相对位置：

```text
spatial-CLAP/
├── pkgs/                         # Linux CPython3.10 wheels + hash lock
└── offline_teacher/
    ├── install_offline.py
    ├── spatial_teacher.py
    ├── check_teacher.py
    ├── extract_features.py
    ├── smoke_fixture.npz
    ├── teacher_provenance.json
    ├── bundle_manifest.json
    ├── weights/spatial_encoder.pt
    └── .venv/                    # 在目标 Linux 机运行安装脚本后生成
```

项目原有根目录 `.venv` 是旧 Windows 环境，不属于本包，也不要复制到 Linux 使用。
`.venv` 不可跨 Windows/Linux 直接搬运。这里使用本地 wheel 在最终路径创建 Linux 环境，之后激活使用。
不额外下载 Python 解释器，使用离线机已有的 Python3.10；需具备标准库 venv 模块。

## 完全离线安装

在离线机的项目根目录执行：

```bash
python3.10 offline_teacher/install_offline.py
source offline_teacher/.venv/bin/activate
```

脚本逐文件验证 SHA256，然后在 `offline_teacher/.venv` 创建独立环境。
pip 也从本地 wheel 引导安装；使用 `--no-index --require-hashes`，禁用外部索引、缓存和用户包。
安装后执行 `pip check`、CUDA 真实前向、原始/交换/重复输入检查，并与本地官方模型参考特征比较。
成功结果写到 `offline_teacher/installation_check.json`。GPU 不可用时明确失败，不悄悄转 CPU。

只检查包完整性：

```bash
python3.10 offline_teacher/install_offline.py --verify-only
```

仅在无 GPU 机器进行 CPU 诊断时可以指定 `--device cpu`；这不算通过目标 GPU 验证。

如果 `glibc < 2.28`，这套 CUDA13 PyTorch wheel 不兼容，需使用事先准备好的合适 Linux/容器环境。
系统显示 CUDA13.0 不等价于实际 GPU 运算一定可用；最终以 `check_teacher.py` 的真实前向为准。
脚本会输出 GPU 实际名称、compute capability、PyTorch 内置 CUDA 架构及运行库版本。
如 NVIDIA M402 是云平台别名，本包不假定其等同于某一零售型号。

## 使用教师

```python
import sys
sys.path.insert(0, "/absolute/path/spatial-CLAP/offline_teacher")
from spatial_teacher import SpatialTeacher

teacher = SpatialTeacher().to("cuda")
# wave: [B, 2, N]，16 kHz float PCM，两侧保持原始相对音量。
S = teacher(wave)  # [B,256]，L2 归一化、冻结、no_grad、float32
```

输入 4096 samples 对应之前测试的 256 ms 窗口。不要分别归一化左右声道；不要按声道单独自动增益。
原始波形超过 512 samples 即可前向，但本包经过验证的窗口为 4096 samples。
模型默认关闭 autocast，避免 DiT 的混合精度上下文改变教师目标。
本包仅提供教师前向/离线提取，不修改 DiT/VAE，也不实现学生损失。

提取 WAV/FLAC/视频中的全部非重叠完整窗口：

```bash
offline_teacher/.venv/bin/python offline_teacher/extract_features.py input.wav --output features.npz
offline_teacher/.venv/bin/python offline_teacher/extract_features.py input.mp4 --output features.npz
```

NPY 输入要求 `[samples,2]`，另加 `--sample-rate 16000`。
音频非16kHz时用 SciPy 有理数重采样。视频通过 imageio-ffmpeg wheel 内置的 Linux FFmpeg 解码；不依赖项目旧 Windows ffmpeg.exe。
保持原始声道数；单声道输入拒绝，不自动复制为双声道。尾部不足256ms的窗口忽略。
输出含 `[windows,256]` features 和 start/end_seconds。
这里不自动过滤居中/高相似度音频；有效负样本的筛选应在学生训练数据预处理中进行。

## 模型与验证范围

来自 https://github.com/sarulab-speech/SpatialCLAP 的独立 spatial 子模块；提取原有已验证 checkpoint 的40项 state。
权重无训练、无量化，紧凑 checkpoint 约4.4MB。模块层结构/state键与官方一致；本地 CPU 参考对比误差记录于 teacher_provenance.json。
包不包含完整 CLAP 内容编码器/文本编码器的运行依赖，所以原项目旧的完整 CLAP 评测脚本不承诺在此精简环境运行。

准备机是 Windows，无法直接执行 Linux ELF wheel。已校验 wheel 平台、Python ABI、完整 Linux 依赖及 SHA256；
目标 GPU/CUDA13 的最终验证由离线机安装末尾的 smoke test 完成，不能把本机 CPU 验证当成目标 GPU 验证。

官方 PyTorch 版本安装参考：https://pytorch.org/get-started/previous-versions/

## 训练数据预处理：筛选 → 三组 PT 表征

新增两个主要入口：`filter_spatial_audio.py` 和 `infer_spatial_features.py`，共享 `spatial_pipeline.py`。
它们都在终端输出进度，并把完整参数、FFmpeg命令、窗口统计和异常写入项目 `running_log/`。
以下命令均在项目根目录执行，使用前面安装好的环境。

### 1. 筛选输入目录

```bash
offline_teacher/.venv/bin/python offline_teacher/filter_spatial_audio.py \
  --input-dir /data/videos \
  --chunk-len 256 \
  --min-ild-db 0.5 \
  --min-total-dbfs -45
```

默认输出目录为项目根目录下 `output_videos`，可以用 `--output-dir /data/screened` 覆盖。
目录输入只扫描当前一级支持的视频文件，不递归。

TXT 输入（每行一个视频路径，不需要加引号；相对路径按 TXT 文件所在目录解析）：

```bash
offline_teacher/.venv/bin/python offline_teacher/filter_spatial_audio.py \
  --input-txt /data/selected.txt --input-dir /data/ignored \
  --output-dir /data/screened --chunk-len 256
```

同时传入 TXT 和目录时，TXT 优先；TXT 中的视频要求来自同一目录。
重复路径去重，同名不同扩展名造成文件名冲突时会明确报错。
筛选阶段无需 GPU，也不加载教师。

窗口规则：

- 提取第一条音轨，重采样16kHz，保留原始声道数；拒绝单声道/多于两声道输入，不做上混。
- 最多保留前10060ms；`num_chunk = ceil(10060/chunk_len)`，窗口长度固定，单位毫秒。
- 不足部分重复最后一个完整源窗口；短于一窗时重复已有音频。任何部分/全部 padding 的窗口都设 `valid_mask=false`。
- `E_L=mean(L²)`、`E_R=mean(R²)`；能量阈值为 `10log10((E_L+E_R)/2) >= min_total_dbfs`。
- 左右差阈值为 `abs(10log10((E_L+1e-12)/(E_R+1e-12))) >= min_ild_db`。
- 默认阈值分别为 −45 dBFS、0.5 dB；仅用这两项阈值和完整窗口条件，不额外过滤左右波形相似度、活动率等。
- 所有窗口默认256ms，`num_chunk=40`；10.06s下最多39个完整窗口，最后一个设 `valid_mask=false`。
- 教师要求每窗超过512采样点，所以 `chunk_len` 最小33ms；建议沿用已验证的256ms。

输出：

```text
<output-dir>/
├── filtered.jsonl
├── filter_summary.json
└── <视频stem>_<窗口索引>.mp4
```

JSONL 每行一个至少有一窗有效的视频，例如：

```json
{"video":"/data/videos/a.mp4","num_chunk":4,"chunk_len":3000,"valid_mask":[true,false,true,false]}
```

上例展示3000ms窗口的4项序列；默认256ms时真实文件严格保存40项。还包含 sample_rate、source_num_samples、real_sample_counts、筛选阈值/能量等元数据。
`video` 是当前运行机器上的绝对路径；如把已生成的JSONL搬到不同数据路径，需要重新筛选生成或更新其中路径。
源时间以解码后的音轨起点为0，音视频本身应已对齐。

从所有保留视频的有效窗口中全局等概率、无放回抽样，默认最多100个；不足100个则全部导出。
可用 `--sample-count 0` 关闭导出，用 `--seed 42` 固定抽样。
MP4编码时长受视频帧率/AAC包长度量化影响，听看样例可能有少量边界舍入；实际筛选和教师计算按精确采样点切窗。

### 2. 提取三组表征

```bash
offline_teacher/.venv/bin/python offline_teacher/infer_spatial_features.py \
  --input-jsonl /data/screened/filtered.jsonl \
  --output-dir /data/features --device cuda --batch-size 32
```

不接受 TXT；TXT 必须先经过筛选脚本。JSONL 中的视频要求来自同一个目录。
默认 `output-dir` 为视频目录的父目录，例如视频在 `/data/videos`，默认输出父目录为 `/data`。

```text
<output-dir>/
├── videos_S+/a.pt
├── videos_S_exc/a.pt
├── videos_S_mono/a.pt
└── videos_spatial_inference_summary.json
```

每个视频每种变换只输出一个 PT，字段包括：

- `features`: float32 `[num_chunk,256]`，只有 `valid_mask=true` 的行经过教师并L2归一化，其余行精确为零。
- `valid_mask`: bool `[num_chunk]`，三个文件完全一致。
- `video`、`num_chunk`、`chunk_len`、`sample_rate`、`num_samples_per_chunk`、`source_num_samples`。
- `start_seconds`、`end_seconds`: float64 `[num_chunk]`，是固定时间网格；padding区间会超过实际音频长度。
- `real_sample_counts`: int64 `[num_chunk]`，标记每窗真实采样点数。
- `transform`、`teacher_checkpoint_sha256`、`screening` 等。

三种变换不改变窗口索引：

- `S+`：原双声道。
- `S_exc`：交换L/R。
- `S_mono`：每个采样点有符号 `M[n]=max(L[n],R[n])`，复制成 `[M,M]`。

`S_mono` 不使用绝对值、不做整窗强侧选择，也不补偿增益；逐点max会改变频谱/直流成分，不能视为严格语义不变变换。
脚本跳过所有 `valid_mask=false` 窗口的教师计算，用零向量保留时间轴。
后续训练必须用 `valid_mask` 同时屏蔽损失和负样本池，不能把零向量当教师目标。

默认 GPU，GPU 不可用时明确失败；本地调试可显式 `--device cpu`。
可用 `--checkpoint` 指定空间模块/完整音频 checkpoint，默认使用本包已验证的精简权重。

### 重跑和错误定位

输出目录存在时正常复用；同名 JSONL/PT/MP4 已存在时默认拒绝覆盖，需要显式 `--overwrite`。
不会清空目录或删除其他文件。PT和JSONL均先写临时文件再原子替换。
单个媒体失败时记录错误并继续处理其余视频；失败列表在summary和日志中。
退出码0表示成功；1表示全局配置/输入错误；2表示部分处理失败、样例导出失败或没有可输出的有效视频。
可用 `--verbose` 同时在终端查看FFmpeg参数及逐窗/批次细节；日志文件默认始终保留这些细节。

### 3. 检查正负表征的余弦差异

```bash
offline_teacher/.venv/bin/python offline_teacher/evaluate_teacher_similarity.py \
  --output-dir /data/features
```

传入上一阶段表征推理的 `--output-dir`，自动扫描其直接子目录 `<目录名>_S+`、`_S_exc`、`_S_mono`，按目录前缀和同名PT匹配。
支持同一个 output-dir 中包含多组目录前缀；每组内匹配，最后全局汇总。
启动时先打印 `+/exc: xxx total, xxx matched` 和 `+/mono: ...`；total 是正样本PT文件数，matched是文件名可匹配的对数。
同时打印负样本总数以及仅一侧存在的文件数。匹配对数和有效窗口数是不同单位。

逐文件读取时检查窗口数、`valid_mask`、时间轴、教师hash等是否一致。只对 `valid_mask=true` 的相同索引逐行计算余弦；
不把整个视频flatten成一个向量，不统计零占位，不按视频先取均值。
结果按所有有效窗口等权汇总，分别给出mean、median、p90、p95，不逐条输出余弦。
两份mask不一致、有效行非有限/零向量等属于错误，该文件对不计入统计，并在总统计和 running_log 里记录。

默认保存 `<output-dir>/teacher_similarity_summary.jsonl`，两行分别为 `S+/S_exc`、`S+/S_mono`，
包含文件匹配数、成功/失败对数、有效窗口数和四项统计。可用 `--output-jsonl /data/stats.jsonl` 指定路径，重跑覆盖需 `--overwrite`。
没有有效窗口时对应统计为JSON的null，退出码2；存在损坏文件时可生成partial统计并返回2。
CPU计算，不需要GPU、原视频或重新运行教师。此处统计的是变换敏感性，不是方位准确率。
