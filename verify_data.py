"""Verify input integrity, label bounds and molecule-disjoint grouped splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parent
TASKS = {'absorption': ('Absorption_Maximum_nm',15195,7316,73,71), 'emission': ('Emission_Maximum_nm',8061,3903,63,62), 'quantum_yield': ('Fluorescence_Quantum_Yield',7033,3334,62,61)}
def main():
    RDLogger.DisableLog('rdApp.*')
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    # Training is allowed to replace result tables. Integrity checks cover source data only.
    for name, expected in manifest.items():
        if name.startswith('trained-model/'): continue
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for task, (column, records, molecules, names, structures) in TASKS.items():
        table = pd.read_csv(ROOT / f'data_filtered_{task}.csv')
        assert (len(table), table.Canonical_SMILES.nunique(), table.standard_solvent.nunique(), table.solvent_smiles.nunique()) == (records,molecules,names,structures), task
        assert table[['Canonical_SMILES','standard_solvent','solvent_smiles',column]].notna().all().all(), task
        assert np.isfinite(table[column]).all(), task
        assert table[column].between(0,1).all() if task == 'quantum_yield' else (table[column] > 0).all(), task
        for smiles in set(table.Canonical_SMILES) | set(table.solvent_smiles):
            assert Chem.MolFromSmiles(smiles) is not None, smiles
        train, test = next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(table, groups=table.Canonical_SMILES))
        assert set(table.iloc[train].Canonical_SMILES).isdisjoint(table.iloc[test].Canonical_SMILES), task
        print(f'{task}: {records} records; valid inputs; molecule-disjoint split')
    collected = pd.read_csv(ROOT / 'evaluation-data/photophysical_records_collected.csv')
    assert len(collected) == 164 and collected.Record_ID.is_unique
    print('Source data checks passed.')
if __name__ == '__main__': main()
