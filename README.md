# Gastric World Model

胃癌治疗条件生成模型与 pCR / 记录性复发预测研究代码。
主线是 Generated V2 的 complete651 五折十种子实验；700 人版本与临床基线保留为重要对照。
源码来自 2026-09-16 的独立实验目录，不包含患者数据、特征、预测、模型权重或运行日志。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[binary-endpoints,dev]'
python run.py gastric scripts/run_generated651.py --help
```

使用 Python 3.11 或 3.12，GPU 版本 PyTorch 需匹配 CUDA。原始 CT 特征准备还需要 `ct` 可选依赖和
按其条款取得的 Swin UNETR 权重；仅从已验证的缓存特征训练不需要重复运行影像编码器。

## 数据准备

主入口读取一个既有 700 人来源 pool，按完整 CT0/CT1 和双终点标签的交集建立 complete651 pool。
实际训练预期 651 人；它不是接收任意 Excel 后自动确定终点的通用工具。
原始临床映射、CT 预处理及缓存构建代码保留在 `src/stageworld/data` 与相关 workflow 中；
字段映射示例在 `configs/`，仅可按研究方已确认的数据与时间定义使用。
外部输入合同与准备入口见 [数据说明](docs/DATA_AND_PATHS.md)。

```bash
python run.py gastric scripts/run_generated651.py --source-pool /path/to/source-pool.pt --pool /path/to/complete651 --output /path/to/study/formal --prepare-only
```

代码路径随仓库提供。需要配置历史路径的辅助工作流使用 `paths.local.yaml`，该文件不会进入 Git。
缺少真实输入或权重时会报错，合成数据只用于显式选择的测试。

## 训练、恢复与评估

```bash
# 短流程仍需真实来源 pool；无数据时使用下方合成测试
python run.py gastric scripts/run_generated651.py --source-pool /path/to/source-pool.pt --pool /path/to/complete651 --output /path/to/study/smoke --smoke

# 正式训练；同一路径再次执行会校验状态并按已有恢复合同继续
python run.py gastric scripts/run_generated651.py --source-pool /path/to/source-pool.pt --pool /path/to/complete651 --output /path/to/study/formal

# 只验证既有结果和独立推理 bundle
python run.py gastric scripts/run_generated651.py --source-pool /path/to/source-pool.pt --pool /path/to/complete651 --output /path/to/study/formal --verify-only

# 700 人对照入口
python run.py gastric scripts/run_generated700.py --help
python run.py gastric scripts/run_binary700.py --help
```

推理接口为 `stageworld.generated651_inference.predict_bundle`；输入仅含可用的基线资料与 CT0，
不读取未来 CT1。模型 bundle 的导出与重放由同模块及 `generated651_verification` 提供。

## 固定的主线协议

- 五折患者级划分，十个模型种子共用这些折：17、43、97、131、173、211、257、307、359、419。
- 三个联合训练分支：BCE、正类加权 BCE、正类加权 focal；外部 Swin 与临床 logistic anchor 冻结。
- 预处理、临床 anchor 和类别权重只在训练折拟合；CT1 仅为生成监督目标。
- **每个种子单独报告五折均值和样本标准差，不把十个种子再平均。**
- 验证折同时选择检查点，结果属于开发验证估计，不是独立测试结论。

完整版本边界见 [实验索引](docs/EXPERIMENTS.md)。原三阶段生存模块作为依赖和历史研究背景保留，
不代表 complete651 主运行路径使用了手术病理或生存目标。

## 合成验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python run.py gastric -m pytest -q tests/test_generated651.py tests/test_generated700.py tests/test_binary700.py tests/test_binary_improvement.py
```

测试覆盖病例交集、折间隔离、训练折拟合、未来输入限制、短训练与恢复以及按种子汇总。
实测状态见 [验证记录](docs/VALIDATION.md)，代码来源和许可见 [来源说明](docs/SOURCES.md)。
