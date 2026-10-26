# Spatial-CLAP 空间教师：离线筛选与表征提取

本仓库包含代码、精简教师权重、合成测试音频和精确依赖清单。
Git 不保存2.71GiB离线wheel、原始数据、缓存、生成表征或虚拟环境。
从 GitHub 获取项目时，先在联网机器恢复wheel，再把整个目录复制到离线开发机：

```bash
git clone https://github.com/yajyuusenpai-syo/cl_sample_create.git
cd cl_sample_create
python download_wheelhouse.py
```

后续安装和所有数据处理均可离线运行。模型及引用来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Linux x86_64 / Python3.10 / CUDA13.0 离线运行说明见 [offline_teacher/README.md](offline_teacher/README.md)。

```bash
# 在项目根目录，用离线机已有的 Python3.10 安装；只使用 pkgs 里的本地 wheel。
python3.10 offline_teacher/install_offline.py
source offline_teacher/.venv/bin/activate

# 目录输入；也可 --input-txt /data/videos.txt，TXT 优先于 --input-dir。
python offline_teacher/filter_spatial_audio.py --input-dir /data/videos \
  --chunk-len 256 --min-ild-db 0.5 --min-total-dbfs -45

# 每个视频/变换保存一个 [num_chunk,256] PT 字典；统一 valid_mask。
python offline_teacher/infer_spatial_features.py \
  --input-jsonl output_videos/filtered.jsonl --device cuda

# 汇总有效窗口的正/交换、正/逐点max双声道余弦；只输出总统计。
python offline_teacher/evaluate_teacher_similarity.py --output-dir /data
```

筛选默认输出到项目 `output_<视频目录名>`；推理默认输出到视频目录父目录下的 `<视频目录名>_S+`、`_S_exc`、`_S_mono`。
两者都支持 `--output-dir`，日志写到项目 `running_log/`。重跑覆盖同名输出需显式 `--overwrite`。
模型为冻结的独立 spatial encoder；padding及筛选未通过的窗口都保持 `valid_mask=false`，PT对应行为零。
`S_mono` 按逐采样点 `max(L,R)` 复制为相同双声道，不同于历史整窗强侧复制实验。

下文保留的是之前的能量/GCC评测入口；新教师数据预处理请使用上面的两个脚本。

# 历史：左右耳能量余弦 / GCC-PHAT 100 ms 汇总评测

当前入口已切换为 `evaluate_spatial_cues.py` 或 `run_validate.ps1`，不再运行 SpatialCLAP。旧权重和旧实验脚本保留，但不参与当前默认评测。

```powershell
cd 'C:\path\to\cl_sample_create'
.\.venv\Scripts\python.exe .\evaluate_spatial_cues.py
.\.venv\Scripts\python.exe .\evaluate_spatial_cues.py --video cs_new1 --out .\outputs\energy_gcc_cs_new1_100ms_summary
```

默认读取同父目录的 `new_100_samp_720p` 视频，每 100 ms 独立评测、不重叠、不进行时序打乱。FFprobe 校验原生双声道，复用或更新 16 kHz 双声道 WAV 缓存。所有窗口相关数据仅在内存中计算。

## 当前筛选

已取消：单侧近静音、近反相单声道、近双单声道三个直接过滤条件。仍计算这些指标作汇总分组，不凭它们删除窗口。

当前只保留整体静音/低活动率及严重削波过滤。方向线索不足也不再过滤，以免间接剔除居中和近双单声道。信号有无可靠线索由有效掩码表达。

- 左右耳能量：各 8 个 100～7500 Hz 频带，频带内 STFT 功率求和、时间平均，拼为 `[E_L,1...E_L,8,E_R,1...E_R,8]` 共 16 维非负向量。左右耳不分别归一化，余弦比较整个拼接向量。另提供 2 维 `[mean(L²),mean(R²)]` 总能量向量的原始/变换余弦。
- GCC-PHAT：保留 ±1 ms 的有符号归一化相关曲线；8 倍插值，257 个延迟点，间隔 7.8125 μs。峰值给出延迟，但多峰和混响情况下不能直接理解为真实单一声源方向。

完全相同 L/R 的交换结果本来不变，应当作为不变性对照。反相和居中音频的交换也不自动构成有效空间负样本。零能量向量的余弦不定义；单侧无声无法估计延迟时仍保留有效能量目标。

## 计算与输出

只使用 NumPy、SciPy 和 SoundFile，在 CPU 上计算，不加载 PyTorch 或任何预训练权重，无 VAE 或 vocoder。

当前只构造两种变换，分别与原始音频比较，不再构造共同增益样本：

- `E_exc`：交换 L/R。
- `E_mono`：对每个已通过质量筛选的 100 ms 窗口，按整窗波形均方能量选择较强的一侧，将其完整波形复制到另一侧，得到 `[X_stronger,X_stronger]`。能量完全相等时选左侧。不按频带分别选边，不补偿音量。

分别报告两种变换相对于原始的 16 维能量余弦、2 维总能量余弦、GCC 余弦和延迟变化。未定义时单独计数，不伪造余弦。不再使用 ILD。

同窗另统计 `cos(E_L,E_R)`，它只反映两耳能量频谱形状是否相似：即使右耳能量是左耳的 0.1 倍，该值仍可为 1。因此交换对比的主要能量指标使用有左右顺序的 16 维拼接向量，保留相对响度关系。

交换后能量向量的左右两半互换，GCC 曲线沿延迟轴翻转。复制后双声道完全一致、延迟应为零，同时总能量可能增加，不做响度补偿。`E_exc` 与 `E_mono` 不预设谁应更相似，不计算正负排序通过率；额外报告二者余弦之差，仅作描述。

能量向量非负，余弦通常位于 0～1。这些变换测试不是定位准确率，能量/GCC 也不包含声音身份或视觉语义对应。

仅保存以下两个文件，运行结束才输出总统计，不打印逐视频、逐窗口或逐样本结果：

- `outputs/energy_gcc_100ms_summary/summary.json`
- `outputs/energy_gcc_100ms_summary/REPORT.md`

DiT 接入建议：离线提取上述目标，训练时从音频隐藏状态或空间残差用两个小头预测能量分布和 GCC。能量目标对整个 16 维向量联合归一化，保留两耳能量比例；使用余弦配合分布/回归损失，GCC 使用曲线回归或延迟分布监督。当前只实现离线提取与验证，尚未修改 DiT。

旧评测输出已按用户要求清理。源码、权重、环境、原视频和原始音频缓存保留。

## 环境

历史 Windows `.venv` 使用 `setup.ps1` 指定的基础 Python 环境，与新的 `offline_teacher/.venv` 相互独立。历史能量/GCC计算不导入或使用 PyTorch；该 `setup.ps1` 只补齐 NumPy、SciPy、SoundFile、Requests 和 FFmpeg，不下载或检查神经网络权重/CUDA。

官方仓库位于 `upstream`，来源：https://github.com/sarulab-speech/SpatialCLAP 。历史实验脚本保留供参考，但默认运行入口已经切换为本汇总评测。
