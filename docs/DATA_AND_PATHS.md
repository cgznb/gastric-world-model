# 数据、路径与复现前提

本仓库发布源码与示例配置。真实影像、临床表、患者列表/划分、缓存、预测和模型权重由研究方独立保管。
测试中的数据由程序构造，不是临床数据的子集。

## 路径配置

将根目录的 `paths.example.yaml` 复制成 `paths.local.yaml`；后者被 Git 忽略。
也可以通过 `RESEARCH_PATHS_CONFIG` 指定另一个本地 YAML 文件。

| 标记 | 含义 |
|---|---|
| `@repo/…` | 本仓库内的源码或配置，自动随克隆位置变化 |
| `@data/…` / `@weights/…` | 通过 `paths.data` / `paths.weights` 指定的输入目录 |
| `@artifacts/…` | 通过 `paths.artifacts` 指定的运行产物目录 |
| `@workspace/…` / `@external/…` | 历史数据布局的外部根目录，默认在仓库 `external/` 下 |
| `@python` | 当前运行 `run.py` 的 Python 解释器 |
| `@legacy:…` | 原数据/权重合同的兼容字段，仅从私有原配置读取 |
| `@private:…` | 私有队列纳入/排除信息，从本地 `cohort` 配置读取 |

不必重建旧目录树：在 `path_overrides` 中将配置里的完整标记字符串映射为对应实际文件或目录即可。
路径可以相对仓库根目录，也可以是绝对路径。不要把本地路径文件加入 Git。

MRI 历史配置中的 `legacy_config_roots` 按源码键填写原配置根目录，例如 `ispy2`、`mu`、`symm`、
`ispy2_codec`。它们只提供输入合同中的身份字段，不作为 Python 源码导入。
MU 固定缺失掩膜记录通过 `cohort.mu_missing_baseline` 与 `cohort.mu_missing_late` 在本地提供；
仓库没有导出原患者标识。导入模块时使用的合成占位符不能满足正式数据验证。

模型结构读取可以只解析结构参数；实际数据/权重加载仍保留原来的合同校验。
没有提供相应私有输入时，真实训练不会退回随机模型或自动获取患者数据。

## MRI 输入

配准 BiFM/SymmFlow 需要与各配置匹配的影像清单、训练/验证划分、VQ 权重和连续 latent 缓存。
first-post/三相位需要原生影像、分割及几何记录、VQ 运行目录和 shared bundle。
ROI32 还需要已有配准变换；变换读取辅助源码包含在 `support/registration`，ITK 环境可独立指定。
远程准备使用本地 `ssh.host`、`ssh.arguments` 和 `paths.remote_data`，不包含原机器的 SSH 配置。
已有缓存若内部保存了绝对路径，应按相同数据合同重新生成可迁移的清单；代码搬迁不会自动重写患者产物。

## 胃癌输入

`run_generated651.py` 的 `--source-pool` 是700人来源缓存，包含有序样本、临床与治疗字段、CT0/CT1
特征及标签可用掩膜；读取合同见 `generated700_data.Pool` 与 `generated651_data.prepare_pool`。
`--pool` 是新生成的完整病例缓存目录，`--output` 是该实验输出目录。
准备来源数据时，先配置 `configs/field_mapping.example.yaml` 和项目 YAML，再使用 `stageworld` 的数据
审计/准备入口与 `scripts/run_coarse_roi_pipeline.py` 等相应阶段工具；可用参数以 `--help` 为准。
终点定义、CT 选择和时间权限需与原协议一致，不能用合成 smoke 替代这些输入确认。

## 依赖边界

主训练的核心依赖列于仓库依赖文件。原始影像准备的可选依赖包括 nnU-Net、ITK/Elastix 和相应权重。
历史 FM-BCMRI 等外部编码器接口可以保留在支持库中，但不因此成为当前 BiFM 或 SymmFlow 的训练输入。
不自动下载模型，不新建远程训练服务，不继承原工作目录的 `PYTHONPATH`。
