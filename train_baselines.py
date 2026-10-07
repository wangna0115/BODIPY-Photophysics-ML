"""Train the four feature combinations and four tree regressors."""
from pathlib import Path
import argparse
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['absorption','emission','quantum_yield'], required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--max-rows', type=int, default=None, help='Smoke checks only; omit for reported experiments.')
    parser.add_argument('--features', nargs='+', choices=['morgan_morgan','rdkit_rdkit','morgan_onehot','rdkit_onehot'], default=['morgan_morgan','rdkit_rdkit','morgan_onehot','rdkit_onehot'])
    parser.add_argument('--models', nargs='+', choices=['CatBoost','XGBoost','RandomForest','ExtraTrees'], default=['CatBoost','XGBoost','RandomForest','ExtraTrees'])
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.jobs < 1 or args.trees < 1 or (args.max_rows is not None and args.max_rows < 10):
        parser.error('jobs/trees must be positive; max-rows must be at least 10')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir.resolve())
    Path('trained-model').mkdir(exist_ok=True)

    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from sklearn.model_selection import GroupShuffleSplit
    from joblib import Parallel, delayed
    from sklearn.preprocessing import OneHotEncoder

    # 可选择 absorption、emission 或 quantum_yield
    task = args.task

    task_columns = {
        'absorption': 'Absorption_Maximum_nm',
        'emission': 'Emission_Maximum_nm',
        'quantum_yield': 'Fluorescence_Quantum_Yield'
    }

    df_raw = pd.read_csv(ROOT / f'data_filtered_{task}.csv')
    if args.max_rows:
        df_raw = df_raw.head(args.max_rows).copy()
    target_column = task_columns[task]
    y = df_raw[target_column].to_numpy(dtype=float)

    # 并行计算需要一个简单函数，每次处理一个分子
    def calculate_substrate_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        morgan = generator.GetFingerprintAsNumPy(mol)
        rdkit_descriptors = Descriptors.CalcMolDescriptors(mol) if any(name.startswith("rdkit_") for name in args.features) else {}
        return morgan, rdkit_descriptors

    # 每个独立结构只计算一次；可直接修改进程数
    n_jobs = args.jobs
    unique_substrate_smiles = df_raw['Canonical_SMILES'].unique()
    substrate_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(calculate_substrate_descriptors)(smiles)
        for smiles in unique_substrate_smiles
    )

    substrate_morgan_cache = {}
    substrate_rdkit_cache = {}
    for smiles, (morgan, rdkit_descriptors) in zip(unique_substrate_smiles, substrate_results):
        substrate_morgan_cache[smiles] = morgan
        substrate_rdkit_cache[smiles] = rdkit_descriptors

    # 按原始数据行顺序组装，保证描述符与标签对应
    substrate_morgan = np.array([
        substrate_morgan_cache[smiles] for smiles in df_raw['Canonical_SMILES']
    ], dtype=np.uint8)

    substrate_rdkit_df = pd.DataFrame([
        substrate_rdkit_cache[smiles] for smiles in df_raw['Canonical_SMILES']
    ], index=df_raw.index)
    substrate_rdkit = substrate_rdkit_df.to_numpy(dtype=float)

    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    solvent_morgan_cache = {}
    solvent_rdkit_cache = {}

    for smiles in df_raw['solvent_smiles'].unique():
        mol = Chem.MolFromSmiles(smiles)
        solvent_morgan_cache[smiles] = morgan_generator.GetFingerprintAsNumPy(mol)
        solvent_rdkit_cache[smiles] = Descriptors.CalcMolDescriptors(mol) if "rdkit_rdkit" in args.features else {}

    solvent_morgan = np.array([
        solvent_morgan_cache[smiles] for smiles in df_raw['solvent_smiles']
    ], dtype=np.uint8)

    solvent_rdkit_df = pd.DataFrame([
        solvent_rdkit_cache[smiles] for smiles in df_raw['solvent_smiles']
    ], index=df_raw.index)
    solvent_rdkit = solvent_rdkit_df.to_numpy(dtype=float)

    # XGBoost 使用 float32，有限但过大的数值也无法输入模型
    float32_limit = np.finfo(np.float32).max
    substrate_values = substrate_rdkit_df.to_numpy(dtype=float)
    solvent_values = solvent_rdkit_df.to_numpy(dtype=float)

    # 分别确定底物和溶剂中需要删除的描述符列
    substrate_valid = np.isfinite(substrate_values) & (np.abs(substrate_values) <= float32_limit)
    solvent_valid = np.isfinite(solvent_values) & (np.abs(solvent_values) <= float32_limit)
    substrate_invalid_columns = substrate_rdkit_df.columns[~substrate_valid.all(axis=0)].tolist()
    solvent_invalid_columns = solvent_rdkit_df.columns[~solvent_valid.all(axis=0)].tolist()

    print('Removed substrate descriptors:', len(substrate_invalid_columns))
    print('Removed solvent descriptors:', len(solvent_invalid_columns))
    print('Substrate:', substrate_invalid_columns)
    print('Solvent:', solvent_invalid_columns)

    substrate_rdkit_df = substrate_rdkit_df.drop(columns=substrate_invalid_columns)
    solvent_rdkit_df = solvent_rdkit_df.drop(columns=solvent_invalid_columns)
    substrate_rdkit = substrate_rdkit_df.to_numpy(dtype=float)
    solvent_rdkit = solvent_rdkit_df.to_numpy(dtype=float)

    row_indices = np.arange(len(df_raw))

    # 按分子分组划分
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        group_split.split(row_indices, groups=df_raw['Canonical_SMILES'])
    )

    # 只用训练集拟合溶剂编码器
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.uint8)
    train_solvents = df_raw.iloc[train_indices][['standard_solvent']]
    test_solvents = df_raw.iloc[test_indices][['standard_solvent']]
    solvent_onehot_train = encoder.fit_transform(train_solvents)
    solvent_onehot_test = encoder.transform(test_solvents)

    substrate_morgan_train = substrate_morgan[train_indices]
    substrate_morgan_test = substrate_morgan[test_indices]
    substrate_rdkit_train = substrate_rdkit[train_indices]
    substrate_rdkit_test = substrate_rdkit[test_indices]
    solvent_morgan_train = solvent_morgan[train_indices]
    solvent_morgan_test = solvent_morgan[test_indices]
    solvent_rdkit_train = solvent_rdkit[train_indices]
    solvent_rdkit_test = solvent_rdkit[test_indices]

    X_train = {
        'morgan_morgan': np.concatenate([substrate_morgan_train, solvent_morgan_train], axis=1),
        'rdkit_rdkit': np.concatenate([substrate_rdkit_train, solvent_rdkit_train], axis=1),
        'morgan_onehot': np.concatenate([substrate_morgan_train, solvent_onehot_train], axis=1),
        'rdkit_onehot': np.concatenate([substrate_rdkit_train, solvent_onehot_train], axis=1)
    }
    X_test = {
        'morgan_morgan': np.concatenate([substrate_morgan_test, solvent_morgan_test], axis=1),
        'rdkit_rdkit': np.concatenate([substrate_rdkit_test, solvent_rdkit_test], axis=1),
        'morgan_onehot': np.concatenate([substrate_morgan_test, solvent_onehot_test], axis=1),
        'rdkit_onehot': np.concatenate([substrate_rdkit_test, solvent_onehot_test], axis=1)
    }

    y_train = y[train_indices]
    y_test = y[test_indices]
    train_row_ids = df_raw.iloc[train_indices]['Row_ID'].to_numpy()
    test_row_ids = df_raw.iloc[test_indices]['Row_ID'].to_numpy()

    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

    random_seed = 42
    # 使用本机全部 16 个逻辑 CPU；电脑卡顿时可调小
    model_n_jobs = args.jobs

    # 禁止自动生成 catboost_info 文件夹
    catboost_model = CatBoostRegressor(
        iterations=args.trees,
        depth=6,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=random_seed,
        thread_count=model_n_jobs,
        verbose=False,
        allow_writing_files=False
    )

    xgboost_model = XGBRegressor(
        n_estimators=args.trees,
        max_depth=6,
        learning_rate=0.05,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=random_seed,
        n_jobs=model_n_jobs
    )

    random_forest_model = RandomForestRegressor(
        n_estimators=args.trees,
        random_state=random_seed,
        n_jobs=model_n_jobs
    )

    # ExtraTrees 使用更随机的切分阈值，与随机森林进行对比
    extra_trees_model = ExtraTreesRegressor(
        n_estimators=args.trees,
        random_state=random_seed,
        n_jobs=model_n_jobs
    )

    models = {
        'CatBoost': catboost_model,
        'XGBoost': xgboost_model,
        'RandomForest': random_forest_model,
        'ExtraTrees': extra_trees_model
    }

    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_absolute_error
    from IPython.display import display
    import joblib

    baseline_rows = []

    # 每行对应原始记录；同一个 CSV 保存所有已完成模型的逐条预测值
    prediction_table = pd.DataFrame({
        'Row_ID': np.concatenate([train_row_ids, test_row_ids]),
        'Dataset': ['Train'] * len(y_train) + ['Test'] * len(y_test),
        'Experimental': np.concatenate([y_train, y_test])
    })

    # 每个特征组合使用独立的模型副本
    for feature_name in args.features:
        for model_name in args.models:
            model_template = models[model_name]
            print(f'Training: {feature_name} + {model_name}', flush=True)
            model = clone(model_template)
            model.fit(X_train[feature_name], y_train)
            y_train_pred = model.predict(X_train[feature_name])
            y_test_pred = model.predict(X_test[feature_name])

            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            baseline_rows.append({
                'Task': task,
                'Features': feature_name,
                'Model': model_name,
                'R2': test_r2,
                'MAE': test_mae
            })

            # 每完成一个模型就更新预测文件，便于随时选择组合绘图
            prediction_column = f'{feature_name}__{model_name}'
            prediction_table[prediction_column] = np.concatenate([y_train_pred, y_test_pred])
            prediction_table.to_csv(f'trained-model/{task}_baseline_predictions.csv', index=False)

            # 同时保存模型和构建新反应特征所需的配置
            model_bundle = {
                'model': model,
                'task': task,
                'target_column': target_column,
                'feature_name': feature_name,
                'morgan_radius': 2,
                'morgan_fp_size': 2048,
                'substrate_rdkit_columns': substrate_rdkit_df.columns.tolist(),
                'solvent_rdkit_columns': solvent_rdkit_df.columns.tolist(),
                'solvent_encoder': encoder,
                'input_columns': ['Canonical_SMILES', 'standard_solvent', 'solvent_smiles']
            }
            model_file = f'trained-model/{task}_{feature_name}_{model_name}.joblib'
            joblib.dump(model_bundle, model_file, compress=3)

            mae_unit = '' if task == 'quantum_yield' else ' nm'
            print(f'Test R2: {test_r2:.4f} | Test MAE: {test_mae:.3f}{mae_unit}', flush=True)
            del model

    # 保存当前任务的全部实验结果
    baseline_results = pd.DataFrame(baseline_rows)
    baseline_results.to_csv(f'trained-model/{task}_baseline_results.csv', index=False)
    display(baseline_results.sort_values('R2', ascending=False))
    # Explicitly record the split and runtime used by this execution.
    assert not set(df_raw.iloc[train_indices]['Canonical_SMILES']) & set(df_raw.iloc[test_indices]['Canonical_SMILES'])
    pd.DataFrame({'Row_ID': df_raw['Row_ID'], 'Dataset': np.where(np.isin(np.arange(len(df_raw)), train_indices), 'Train', 'Test')}).to_csv(f'trained-model/{task}_split.csv', index=False)
    import importlib.metadata as metadata
    import json
    Path(f'trained-model/{task}_run.json').write_text(json.dumps({'task': task, 'seed': 42, 'trees': args.trees, 'jobs': args.jobs, 'max_rows': args.max_rows, 'features': args.features, 'models': args.models, 'versions': {name: metadata.version(name) for name in ['numpy','pandas','rdkit','scikit-learn','catboost','xgboost','joblib']}}, indent=2), encoding='utf8')


if __name__ == "__main__":
    main()
