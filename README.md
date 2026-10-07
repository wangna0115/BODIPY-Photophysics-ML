# 数据驱动的 BODIPY 荧光分子光物理性质预测

本仓库对应附件3作品申报书：预测最大吸收波长、最大发射波长和荧光量子产率，包含数据库清洗、四种特征与四种树模型对比、Chemprop 及收集数据的模型测试。

## 数据与文件

|任务|记录数|独立分子 SMILES 数|标准溶剂名称数|独立溶剂结构数|
|---|---:|---:|---:|---:|
|吸收|15195|7316|73|71|
|发射|8061|3903|63|62|
|量子产率|7033|3334|62|61|

三个任务相互重叠，不应相加。申报书中的溶剂数对应独立溶剂结构。分子数按所提供的 Canonical_SMILES 字符串计数。

- `data.csv`：原始输入快照；`data_filtered_*.csv`：三个任务的清洗数据。
- `clean_data.py` 与 `data-Cleaning-Splitting.ipynb`：历史清洗规则；Notebook 还保留标签、分子结构和溶剂分布分析。
- `train_baselines.py` 与 `baseline.ipynb`：四种特征 × CatBoost、XGBoost、RandomForest、ExtraTrees。
- `train_chemprop.py` 与 `baseline-chemprop.ipynb`：可选分子图模型。
- `trained-model/*.csv`：收到的历史指标和逐条预测，供核对申报书。训练产物可自行重建；模型文件不打包。
- `evaluation-data/`：164 条收集记录、来源审查依据以及模型测试结果。
- `run_model_evaluation.py`：结构检查、训练重合排除、预测及指标。
- 绘图脚本：重建模型热图、申报书统计图和测试集散点图。
- `evaluation-data/README.md`：收集数据的构建规则、模型测试结果与已知限制。

## 环境

建议使用与历史树模型相同的 Python 3.12.14 环境：

```sh
conda env create -f environment.yml
conda activate bodipy
```

也可在 Python 3.12 虚拟环境中执行 `python -m pip install -r requirements.txt`。依赖版本取自恢复的原训练环境，不保证不同版本生成相同模型或描述符。`environment-original-windows.yml` 记录原 Windows 环境，供历史追溯，包含平台和 CUDA 依赖；跨平台优先使用 `environment.yml`。本次未在全新环境中验证安装。

以下命令从仓库根目录执行，按顺序运行。

## 复现数据清洗

```sh
python clean_data.py
```

将重建 `data-History/` 中间数据和三个任务 CSV。原始 `data.csv` 保留。Notebook 分析须先执行此步骤。

## 复现树模型比较

```sh
python train_baselines.py --task absorption
python train_baselines.py --task emission
python train_baselines.py --task quantum_yield
python plot_baseline_results.py
python build_report_figures.py
```

默认每组合 500 棵树、种子 42、2 个工作线程。按 Canonical_SMILES 使用 GroupShuffleSplit，以 20% 分子组作测试；one-hot 仅在训练集拟合。沿用历史算法，RDKit 不合法描述符列在整份任务数据上筛除。每次运行保存模型、预测、指标、划分和环境版本。完整比较共 48 个模型，需要足够内存和磁盘；无需将模型上传 GitHub。

只重建申报书模型测试采用的三个模型，可在上述三个训练命令分别追加 `--features morgan_onehot --models RandomForest`。要保留历史 CSV，训练时添加 `--output-dir outputs/retrained`，然后在模型测试时指定 `--model-dir outputs/retrained/trained-model`。

## 模型测试

```sh
python run_model_evaluation.py
python plot_model_evaluation_scatter.py
```

默认模型来自 `trained-model/`，须先训练；输入和结果统一保存在 `evaluation-data/`，其中运行结果位于 `evaluation-data/results/`。运行时会重新检查骨架、有效标签和训练集结构重合，不能只依赖人工状态标记。未知溶剂按原 one-hot 规则编码为全零，并报告。

## 可选 Chemprop

```sh
python -m pip install -r requirements-chemprop.txt
python train_chemprop.py --tasks quantum_yield --epochs 20 --accelerator cpu
```

申报书比较的是量子产率 Chemprop 结果；保留的 Notebook 也可训练三项任务。历史输出记录为 Chemprop 2.2.3、PyTorch 2.14.0+cu130；Lightning 版本缺失，本次未重新验证此环境或神经网络训练。CPU/GPU 及不同软件版本可能产生不同数值。

需要交互分析时另装 `jupyterlab`，从仓库根目录启动。结构空间分析中选择 UMAP 时另装 `umap-learn`。

## GitHub 上传

上传本目录的清理后文件，或交付的干净 ZIP。`.gitignore` 排除模型权重、清洗中间数据、运行输出、缓存和环境。已去除 Notebook 输出及本机绝对路径。无需上传原申报书、备份或约 1.9 GB 的历史权重。

数据来源、人工核查和许可限制见 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)。本次整理没有上传到远端。
