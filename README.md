# Gastric World Model

胃癌治疗条件生成模型与 pCR / 记录性复发预测研究代码。
当前主线是 **Event Multistage：新辅助治疗、手术、术后化疗的多阶段事件模型**，
使用 complete651 五折十种子协议。此前的 Generated V2／651、700 人版本与临床基线保留为对照。
源码来自 2026-09-16 的独立实验目录，不包含患者数据、特征、预测、模型权重或运行日志。

## Event V2 研究分支（2026-09-17）

本分支新增六字段临床 token、阶段专用瓶颈适配、有界残差门控、
基线/当前状态/变化量读出、四成员参数共享分类头，以及训练集 CT0 遮挡重建。
逐层维度、公式、公开论文与仓库依据见 [详细设计](docs/EVENT_V2_RESEARCH.md)。
旧多阶段实现保留，作为新划分下重新训练的对照；新结构的有效性以实际测试结果为准。

每个种子独立进行患者级联合标签分层，651人分为521训练、65验证、65测试。
十个种子的所有预定模型完成验证集选模后，才执行最终测试；测试集不选模型或种子。
同一队列已参与历史开发，因此这是新实验的内部留出测试，不是外部验证。

```bash
python run.py gastric scripts/run_event_v2.py --help

# 只生成并核查各个种子的划分
python run.py gastric scripts/run_event_v2.py --source-pool /path/to/complete651 --bindings configs/event_bindings.local.json --pool artifacts/event-v2/pool --output artifacts/event-v2/formal --prepare-only

# GPU短验证仅使用训练和验证患者，输出与正式实验分离
python run.py gastric scripts/run_event_v2.py --source-pool /path/to/complete651 --bindings configs/event_bindings.local.json --pool artifacts/event-v2/pool --output artifacts/event-v2/gpu-smoke --smoke

# 默认分别训练旧多阶段和V2，各十个种子，共40个预训练/联合训练阶段
python run.py gastric scripts/run_event_v2.py --source-pool /path/to/complete651 --bindings configs/event_bindings.local.json --pool artifacts/event-v2/pool --output artifacts/event-v2/formal
```

`--families` 可预先指定 `event_v2_no_adapter`、`event_v2_no_mask` 或
`event_v2_single_member` 消融；必须在看测试结果前决定，并使用新的输出目录。
结果保存在 `evaluation/test_metrics.csv` 和 `evaluation/report.md`，逐种子报告。
不按测试结果挑选最佳种子，不混合十个种子的患者预测冒充独立样本。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[binary-endpoints,dev]'
python run.py gastric scripts/run_event_multistage.py --help
```

使用 Python 3.11 或 3.12，GPU 版本 PyTorch 需匹配 CUDA。原始 CT 特征准备还需要 `ct` 可选依赖和
按其条款取得的 Swin UNETR 权重；仅从已验证的缓存特征训练不需要重复运行影像编码器。

## 多阶段模型与训练

冻结 Swin 的 CT0 特征 `[B,27,768]` 与六项临床信息初始化 `[B,27,128]` 状态。
三个事件共享四层条件注意力和空间残差转移，各状态保持相同维度：

```text
CT0 + 临床 -> S0 -> 新辅助治疗 -> S1 -> 手术 -> S2 -> 术后化疗 -> S3
                                  |             |                |
                             CT1特征监督     辅助pCR监督      记录性复发监督
```

新辅助阶段输入四个无周期方案/药名 token；手术、术后化疗使用各自事件标记，
不输入未核实的方案细节。CT1、pCR、复发标签只作监督，日期和周期不进入网络。
两个独立预测头读取 S2 和最终有效状态，没有神经网络内的 logistic anchor。

每个种子、每一折先从头预训练，再加载本折最佳预训练权重做联合训练：

| 阶段 | 损失 | 学习率 | 检查点选择 |
|---|---|---|---|
| 预训练 | CT 特征集合损失 + 0.5 × 普通 pCR BCE | 世界模型及 pCR 头 2e-4 | 验证复合损失最小 |
| 联合训练 | 加权复发 BCE + 0.5 × pCR BCE + 0.1 × CT 损失 | 世界模型 2e-5；两个头 2e-4 | 验证复发 AUPRC 最大 |

AdamW、batch32、BF16、梯度裁剪1；每阶段最多100轮、早停15轮，无学习率调度器。
共50次预训练和50次联合训练。完整维度、事件规则及解释边界见
[多阶段协议](docs/EVENT_MULTISTAGE.md)。

## 数据准备

新入口的 `--source-pool` 是已有 **651 人缓存目录**，包含 `pool.pt` 和 `folds.json`。
如果手头只有700人来源缓存文件，先沿用原入口建立完整病例缓存；此命令不执行训练：

```bash
python run.py gastric scripts/run_generated651.py --source-pool /path/to/source700/pool.pt --pool /path/to/complete651 --output artifacts/prepare651 --prepare-only
```

将 `configs/event_bindings.example.json` 复制为被 Git 忽略的
`configs/event_bindings.local.json`，填写临床表和**生成既有缓存时使用的同一份 HMAC 密钥文件**的路径。
JSON只保存文件路径，不保存密钥内容；这些路径直接交给文件系统，不解析 `@repo` 等路径标记。
新入口复核临床表中的手术、术后化疗和双终点标签，并读取现有新辅助治疗字段：

```bash
python run.py gastric scripts/run_event_multistage.py --source-pool /path/to/complete651 --bindings configs/event_bindings.local.json --pool artifacts/event/pool --output artifacts/event/formal --prepare-only
```

原始临床映射、CT 预处理与缓存准备见 [数据说明](docs/DATA_AND_PATHS.md)。
真实运行要求协议内的651人及固定五折，不能用任意临床表替代；缺少数据或权重会明确报错。

## 训练、恢复与评估

```bash
# 正式训练，需要支持BF16的CUDA设备；此入口没有 --smoke 模式
python run.py gastric scripts/run_event_multistage.py --source-pool /path/to/complete651 --bindings configs/event_bindings.local.json --pool artifacts/event/pool --output artifacts/event/formal

# 只复算既有检查点、预测和推理包；此完整研究校验入口也要求BF16 CUDA
python run.py gastric scripts/run_event_multistage.py --source-pool /path/to/complete651 --bindings configs/event_bindings.local.json --pool artifacts/event/pool --output artifacts/event/formal --verify-only
```

保持同一份源码、输入和输出目录再次执行即可恢复。运行合同绑定源码文件大小及修改时间，
不要将历史运行目录直接切换到新克隆后恢复；原运行应保留原源码，本发布用于新的输出目录。

CPU 独立推理接口是 `stageworld.event_inference.predict_bundle`：接收推理包、六项临床、
无周期治疗字段、`[B,3]`事件状态和`[B,27,768]` CT0特征，不读取CT1或终点标签。
训练自动导出每折推理包，并在 `evaluation/` 写入12项终点指标、CT生成对照和各个种子的五折汇总。

## 固定的主线协议

- 五折患者级划分，十个模型种子共用这些折：17、43、97、131、173、211、257、307、359、419。
- 预处理、CT统计和类别权重只在训练折拟合；外部Swin冻结，联合训练更新世界模型和预测头。
- pCR为手术状态辅助目标，复发只读取最后有效状态；普通临床/治疗/事件逻辑回归是独立参照。
- **每个种子单独报告五折均值和样本标准差，不把十个种子再平均。**
- 验证折同时选择检查点，结果属于开发验证估计，不是独立测试结论。
- 固定阈值0.5，加权输出未经校准；复发终点是记录状态，不是指定时间窗内的新发风险。
- 本队列手术及术后化疗事件均记录为已发生，不能据此识别其有无治疗的因果效果。

## 保留的对照入口

```bash
python run.py gastric scripts/run_generated651.py --help
python run.py gastric scripts/run_generated700.py --help
python run.py gastric scripts/run_binary700.py --help
```

Generated V2／651保留原来的BCE、加权BCE、focal三分支和临床logistic anchor。
历史观察更新、生存模块与当前事件模型是不同路径，完整版本边界见 [实验索引](docs/EXPERIMENTS.md)。

## 合成验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python run.py gastric -m pytest -q tests/test_event_multistage.py tests/test_generated651.py tests/test_generated700.py tests/test_binary700.py tests/test_binary_improvement.py tests_release
```

测试覆盖事件跳过、前缀不变性、梯度路由、缺失目标、训练折变换、精确恢复、独立推理及五折汇总。
实测状态见 [验证记录](docs/VALIDATION.md)，代码来源和许可见 [来源说明](docs/SOURCES.md)。
