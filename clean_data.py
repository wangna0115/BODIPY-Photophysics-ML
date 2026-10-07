"""Reproduce the historical cleaning algorithm without notebook state."""
from pathlib import Path
import argparse
import os
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir.resolve())
    Path('data-History').mkdir(exist_ok=True)

    import pandas as pd
    from IPython.display import display

    # 读取原始数据。
    df_raw = pd.read_csv(ROOT / 'data.csv')
    missing = df_raw.isna() | df_raw.apply(
        lambda col: col.map(lambda x: isinstance(x, str) and x.strip() == '')
    )
    print('Missing values by column:')
    print(missing.sum())

    missing_rows = missing.any(axis=1)

    # 将缺失值和纯空白字符串视为缺失。
    missing_canonical = df_raw['Canonical_SMILES'].isna() | (
        df_raw['Canonical_SMILES'].str.strip() == ''
    )

    # 保留原始数据，分别保存筛选后的记录和删除的记录。
    df_canonical_filtered = df_raw.loc[~missing_canonical].copy()
    df_no_canonical = df_raw.loc[missing_canonical].copy()

    print(f'Original records: {len(df_raw)}')
    print(f'Retained records: {len(df_canonical_filtered)}')
    print(f'Removed records: {len(df_no_canonical)}')

    # 保存第 1 轮筛选结果。
    df_canonical_filtered.to_csv('data-History/data_f1.csv', index=False)


    from rdkit import Chem

    # 读取 Canonical SMILES 缺失筛选后的数据。
    df_smiles_check = pd.read_csv('data-History/data_f1.csv')


    # 分别记录两列中 RDKit 无法解析的 SMILES。
    invalid_smiles = {}

    for column in ['SMILES', 'Canonical_SMILES']:
        invalid_rows = []

        for index, smiles in df_smiles_check[column].items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_rows.append(index)

        invalid_smiles[column] = df_smiles_check.loc[invalid_rows].copy()

    print(f'Total records: {len(df_smiles_check)}')
    print('SMILES Check Summary:')
    for column in ['SMILES', 'Canonical_SMILES']:
        print(f'{column}: Invalid records: {len(invalid_smiles[column])}')


    # 按列展示无效记录，保留 CSV 行索引和原始 ID。
    for column in ['SMILES', 'Canonical_SMILES']:
        print(f'\n{column}: Invalid SMILES')
        print('-' * 60)
        with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
            display(invalid_smiles[column][['Row_ID', 'DOI', column]])


    # 按项目规则，只删除含 [H+] 的记录。
    has_suspicious_hydrogen = df_smiles_check['Canonical_SMILES'].str.contains(
        '[H+]', regex=False, na=False
    )

    df_removed_smiles = df_smiles_check.loc[has_suspicious_hydrogen].copy()
    df_smiles_filtered = df_smiles_check.loc[~has_suspicious_hydrogen].copy()

    print(f'Removed records: {len(df_removed_smiles)}')
    print(f'Retained records: {len(df_smiles_filtered)}')

    print('\nRemoved SMILES Records')
    print('-' * 60)
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(df_removed_smiles[['Row_ID', 'DOI', 'Canonical_SMILES']])


    # 保存第 2.1 轮筛选结果。
    df_smiles_filtered.to_csv('data-History/data_f1_f2-1.csv', index=False)


    # 闭环键也使用 ~，兼容不同共振式中的单键和双键。
    bodipy_smarts = (
        '[*]~[#5-]1(~[*])~[#7]2'
        '~[#6,#7]~[#6,#7]~[#6,#7]~[#6,#7]~2'
        '~[#6,#7]~[#6,#7]3~[#6,#7]~[#6,#7]~[#6,#7]~[#7]~1~3'
    )
    bodipy_core = Chem.MolFromSmarts(bodipy_smarts)


    # 分别记录 B/F 条件与骨架匹配结果，保留原数据和索引。
    has_b_and_two_f_rows = []
    bodipy_rows = []

    for smiles in df_smiles_filtered['Canonical_SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        has_b_and_two_f = 5 in atom_numbers and atom_numbers.count(9) >= 2

        has_b_and_two_f_rows.append(has_b_and_two_f)
        bodipy_rows.append(has_b_and_two_f and mol.HasSubstructMatch(bodipy_core))

    has_b_and_two_f_rows = pd.Series(has_b_and_two_f_rows, index=df_smiles_filtered.index)
    bodipy_rows = pd.Series(bodipy_rows, index=df_smiles_filtered.index)

    df_removed_bf = df_smiles_filtered.loc[~has_b_and_two_f_rows].copy()
    df_removed_core = df_smiles_filtered.loc[has_b_and_two_f_rows & ~bodipy_rows].copy()
    df_removed_bodipy = df_smiles_filtered.loc[~bodipy_rows].copy()
    df_bodipy_filtered = df_smiles_filtered.loc[bodipy_rows].copy()


    print('BODIPY Check Summary:')
    print(f'Input records: {len(df_smiles_filtered)}')
    print(f'No B or fewer than two F: {len(df_removed_bf)}')
    print(f'B/F present but no SMARTS match: {len(df_removed_core)}')
    print(f'Total removed records: {len(df_removed_bodipy)}')
    print(f'Retained records: {len(df_bodipy_filtered)}')

    print('\nNo B or Fewer Than Two F')
    print('-' * 60)
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(df_removed_bf[['Row_ID', 'DOI', 'Canonical_SMILES']])

    print('\nB/F Present but No SMARTS Match')
    print('-' * 60)
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(df_removed_core[['Row_ID', 'DOI', 'Canonical_SMILES']])


    # 保存第 2.2 轮筛选结果。
    df_bodipy_filtered.to_csv('data-History/data_f1_f2-1_f2-2.csv', index=False)


    # 明确的单一分子名称与 SMILES 对照；未列入的名称留待核查。
    # 这里识别分子连接结构，不补全名称未指定的立体化学。
    solvent_smiles = {
        'water': 'O',
        'chloroform': 'ClC(Cl)Cl',
        'dichloromethane': 'ClCCl',
        'dimethyl sulfoxide': 'CS(=O)C',
        '1,2-dichloroethane': 'ClCCCl',
        'ethanol': 'CCO',
        'methanol': 'CO',
        'acetonitrile': 'CC#N',
        'toluene': 'Cc1ccccc1',
        'tetrahydrofuran': 'C1CCOC1',
        'n-hexane': 'CCCCCC',
        'diisopropyl ether': 'CC(C)OC(C)C',
        'diethyl ether': 'CCOCC',
        'cyclohexane': 'C1CCCCC1',
        'n,n-dimethylformamide': 'CN(C)C=O',
        'methyl acetate': 'CC(=O)OC',
        'ethyl acetate': 'CC(=O)OCC',
        'ethylene glycol': 'OCCO',
        'glycerol': 'OCC(O)CO',
        'acetone': 'CC(=O)C',
        'dibutyl ether': 'CCCCOCCCC',
        '1,4-dioxane': 'O1CCOCC1',
        'chlorobenzene': 'Clc1ccccc1',
        '2-propanol': 'CC(O)C',
        '1-butanol': 'CCCCO',
        '1-pentanol': 'CCCCCO',
        '1-octanol': 'CCCCCCCCO',
        'cyclohexanone': 'O=C1CCCCC1',
        'tetrachloromethane': 'ClC(Cl)(Cl)Cl',
        'chloromethane': 'CCl',
        '2-methyltetrahydrofuran': 'CC1CCCO1',
        '1-propanol': 'CCCO',
        '1-decanol': 'CCCCCCCCCCO',
        'heptane': 'CCCCCCC',
        'pyridine': 'n1ccccc1',
        'benzonitrile': 'N#Cc1ccccc1',
        'tert-butyl methyl ether': 'COC(C)(C)C',
        'o-xylene': 'Cc1ccccc1C',
        'p-xylene': 'CC1=CC=C(C)C=C1',
        'isopropanol': 'CC(O)C',
        'methylcyclohexane': 'CC1CCCCC1',
        'butanenitrile': 'CCCC#N',
        'propanenitrile': 'CCC#N',
        '2,2,2-trifluoroethanol': 'OCC(F)(F)F',
        'pinacolone': 'CC(=O)C(C)(C)C',
        'n,n-dimethylacetamide': 'CC(=O)N(C)C',
        'anisole': 'COc1ccccc1',
        'benzene': 'c1ccccc1',
        '2-methylbutane': 'CC(C)CC',
        'isooctane': 'CC(C)CC(C)(C)C',
        '1-methylpyrrolidin-2-one': 'CN1CCCC1=O',
        '2-phenoxyethanol': 'OCCOc1ccccc1',
        '1-hexanol': 'CCCCCCO',
        'acetic acid': 'CC(=O)O',
        'pentane': 'CCCCC',
        '2,2,4-trimethylpentane': 'CC(C)CC(C)(C)C',
        'n-methylformamide': 'CNC=O',
        '3-methyl-1-butanol': 'CC(C)CCO',
        '2-butanone': 'CCC(=O)C',
        'trifluoroacetic acid': 'O=C(O)C(F)(F)F',
        'cyclohexene': 'C1=CCCCC1',
        'octane': 'CCCCCCCC',
        'butyl acetate': 'CC(=O)OCCCC',
        'methyl formate': 'COC=O',
        'perchloric acid': 'OCl(=O)(=O)=O',
        '2-methoxyethan-1-ol': 'COCCO',
        '1,2-dichlorobenzene': 'Clc1ccccc1Cl',
        '1-methylsulfinyl-2-propanone': 'CS(=O)CC(=O)C',
        '1,2-dimethoxyethane': 'COCCOC',
        '5,5-dimethyl-1,3-cyclohexadiene': 'C1=CC=CC(C)(C)C1',
        'tetrachloroethylene': 'ClC(Cl)=C(Cl)Cl',
        'trifluoromethanol': 'OC(F)(F)F',
        'tert-butanol': 'CC(C)(C)O',
        '2-pentanone': 'CC(=O)CCC',
        'diethylamine': 'CCNCC',
    }

    # 名称映射包含项目指定的近似约定，仅用于检查。
    solvent_aliases = {
        # 按项目约定，以正己烷和对二甲苯作为代表结构。
        'hexanes': 'n-hexane',
        'xylene': 'p-xylene',
        # 按项目约定解释为邻二氯苯，原名称本身未指定取代位置。
        'dichlorobenzene': '1,2-dichlorobenzene',
        #一些常见的溶剂别名处理
        'degassed cyclohexane': 'cyclohexane',
        'degassed dichloromethane': 'dichloromethane',
        'degassed acetonitrile': 'acetonitrile',
        'anhydrous ethanol': 'ethanol',
        'anhydrous methanol': 'methanol',
        'degassed anhydrous dichloromethane': 'dichloromethane',
        'non-degassed dichloromethane': 'dichloromethane',
        'deionized water': 'water',
    }


    # 吸收溶剂记录
    review_absorption_rows = []

    for index, solvent in df_bodipy_filtered['Absorption_Solvent'].items():
        # 缺失溶剂留到本模块的完整性筛选处理。
        if pd.isna(solvent):
            continue

        # 只规范匹配用的名称，不修改原始溶剂列。
        solvent_name = solvent.lower()
        solvent_name = ' '.join(solvent_name.split())
        if solvent_name == '':
            continue

        solvent_name = solvent_aliases.get(solvent_name, solvent_name) # 使用别名替换（将第一个solvent_name在字典中查询），若合法则保持原名（第二个solvent_name的意思）
        if solvent_name not in solvent_smiles:
            review_absorption_rows.append(index)

    review_absorption_solvents = df_bodipy_filtered.loc[review_absorption_rows].copy()


    # 发射溶剂记录
    review_emission_rows = []

    for index, solvent in df_bodipy_filtered['Emission_Solvent'].items():
        # 缺失溶剂留到本模块的完整性筛选处理。
        if pd.isna(solvent):
            continue

        # 只规范匹配用的名称，不修改原始溶剂列。
        solvent_name = solvent.lower()
        solvent_name = ' '.join(solvent_name.split())
        if solvent_name == '':
            continue

        solvent_name = solvent_aliases.get(solvent_name, solvent_name)
        if solvent_name not in solvent_smiles:
            review_emission_rows.append(index)

    review_emission_solvents = df_bodipy_filtered.loc[review_emission_rows].copy()


    print('Solvent Check Summary:')
    print(f'Absorption solvent records requiring review: {len(review_absorption_rows)}')
    print(f'Emission solvent records requiring review: {len(review_emission_rows)}')


    print('\nAbsorption: Solvent Names Requiring Review')
    print('-' * 60)
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(review_absorption_solvents[[ 'Row_ID', 'DOI', 'Absorption_Solvent']])


    print('\nEmission: Solvent Names Requiring Review')
    print('-' * 60)
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(review_emission_solvents[['Row_ID', 'DOI', 'Emission_Solvent']])


    # 在清洗副本中只清空无法识别的溶剂，保留另一侧可用的数据。
    df_solvent_checked = df_bodipy_filtered.copy()
    df_solvent_checked.loc[review_absorption_rows, 'Absorption_Solvent'] = pd.NA
    df_solvent_checked.loc[review_emission_rows, 'Emission_Solvent'] = pd.NA

    absorption_solvent = df_solvent_checked['Absorption_Solvent']
    missing_absorption_solvent = absorption_solvent.isna() | (absorption_solvent.str.strip() == '')

    emission_solvent = df_solvent_checked['Emission_Solvent']
    missing_emission_solvent = emission_solvent.isna() | (emission_solvent.str.strip() == '')

    # 两侧都没有可用溶剂时，才删除整条记录。
    missing_both_solvents = missing_absorption_solvent & missing_emission_solvent
    df_removed_solvent = df_bodipy_filtered.loc[missing_both_solvents].copy()
    df_solvent_filtered = df_solvent_checked.loc[~missing_both_solvents].copy()

    print(f'Removed records without either usable solvent: {len(df_removed_solvent)}')
    print(f'Retained records: {len(df_solvent_filtered)}')

    df_solvent_filtered.to_csv('data-History/data_f1_f2-1_f2-2_f2-3.csv', index=False)


    # 合并各轮删除记录，最后一列注明首次删除轮次。
    removed_f1 = df_no_canonical.copy()
    removed_f1['Removed_stage'] = 'f1'

    removed_f2_1 = df_removed_smiles.copy()
    removed_f2_1['Removed_stage'] = 'f2-1'

    removed_f2_2 = df_removed_bodipy.copy()
    removed_f2_2['Removed_stage'] = 'f2-2'

    removed_f2_3 = df_removed_solvent.copy()
    removed_f2_3['Removed_stage'] = 'f2-3'

    df_removed_all = pd.concat([
        removed_f1, removed_f2_1, removed_f2_2, removed_f2_3
    ])
    df_removed_all.to_csv('data-History/data_removed.csv', index=False)


    # 读取全部清洗后的最终数据。
    df_split = pd.read_csv('data-History/data_f1_f2-1_f2-2_f2-3.csv')


    # 吸收任务要求波长和吸收溶剂都完整。
    has_absorption = df_split['Absorption_Maximum_nm'].notna()
    absorption_solvent = df_split['Absorption_Solvent']
    has_absorption_solvent = absorption_solvent.notna() & (absorption_solvent.str.strip() != '')

    absorption_rows = has_absorption & has_absorption_solvent
    df_absorption = df_split.loc[absorption_rows].copy()

    # 沿用此前约定，波长去掉小数部分。
    df_absorption['Absorption_Maximum_nm'] = df_absorption['Absorption_Maximum_nm'].astype(int)

    # 先统一名称，再解析别名；没有别名的名称保持不变。
    standard_solvent = df_absorption['Absorption_Solvent'].str.lower()
    standard_solvent = standard_solvent.str.strip()
    standard_solvent = standard_solvent.str.replace(r'\s+', ' ', regex=True)
    standard_solvent = standard_solvent.replace(solvent_aliases)

    # 保存标准名称和对应 SMILES，保留原始溶剂列。
    df_absorption['standard_solvent'] = standard_solvent
    df_absorption['solvent_smiles'] = standard_solvent.map(solvent_smiles)


    # 发射任务要求波长和发射溶剂都完整。
    has_emission = df_split['Emission_Maximum_nm'].notna()
    emission_solvent = df_split['Emission_Solvent']
    has_emission_solvent = emission_solvent.notna() & (emission_solvent.str.strip() != '')

    emission_rows = has_emission & has_emission_solvent
    df_emission = df_split.loc[emission_rows].copy()
    df_emission['Emission_Maximum_nm'] = df_emission['Emission_Maximum_nm'].astype(int)

    # 先统一名称，再解析别名；没有别名的名称保持不变。
    standard_solvent = df_emission['Emission_Solvent'].str.lower()
    standard_solvent = standard_solvent.str.strip()
    standard_solvent = standard_solvent.str.replace(r'\s+', ' ', regex=True)
    standard_solvent = standard_solvent.replace(solvent_aliases)

    # 保存标准名称和对应 SMILES，保留原始溶剂列。
    df_emission['standard_solvent'] = standard_solvent
    df_emission['solvent_smiles'] = standard_solvent.map(solvent_smiles)


    # 量子产率必须有值，保留原有小数。
    has_quantum_yield = df_split['Fluorescence_Quantum_Yield'].notna()
    df_quantum_yield = df_split.loc[has_quantum_yield].copy()

    # 优先采用发射溶剂，缺失时采用吸收溶剂，不覆盖原始列。
    df_quantum_yield['Solvent'] = df_quantum_yield['Emission_Solvent']
    use_absorption_solvent = has_quantum_yield & ~has_emission_solvent & has_absorption_solvent
    fallback_indices = df_split.index[use_absorption_solvent]
    df_quantum_yield.loc[fallback_indices, 'Solvent'] = df_split.loc[fallback_indices, 'Absorption_Solvent']

    # 先统一名称，再解析别名；没有别名的名称保持不变。
    standard_solvent = df_quantum_yield['Solvent'].str.lower()
    standard_solvent = standard_solvent.str.strip()
    standard_solvent = standard_solvent.str.replace(r'\s+', ' ', regex=True)
    standard_solvent = standard_solvent.replace(solvent_aliases)

    # 保存标准名称和对应 SMILES，保留原始溶剂列。
    df_quantum_yield['standard_solvent'] = standard_solvent
    df_quantum_yield['solvent_smiles'] = standard_solvent.map(solvent_smiles)


    print('Dataset Split Summary:')
    print(f'Absorption records: {len(df_absorption)}')
    print(f'Emission records: {len(df_emission)}')
    print(f'Fluorescence quantum yield records: {len(df_quantum_yield)}')


    # 保存三个任务数据集，保留原始列、标准溶剂和溶剂 SMILES。
    df_absorption.to_csv('data_filtered_absorption.csv', index=False)
    df_emission.to_csv('data_filtered_emission.csv', index=False)
    df_quantum_yield.to_csv('data_filtered_quantum_yield.csv', index=False)


if __name__ == "__main__":
    main()
