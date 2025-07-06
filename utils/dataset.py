from datasets import load_dataset
from typing import Tuple
from torch.utils.data import Dataset
from utils.data_utils import str2inputs
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd 

ref_file = load_dataset('ICML2022/ProteinGym', data_files='ProteinGym_reference_file_substitutions.csv', split='train')

class ProteinData(Dataset):
    # put filename without .csv
    def __init__(self, name, num_cycle, data):
        self.name = name
        self.target_seq = ref_file['target_seq'][ref_file['DMS_id'].index(name)]
        self.data = data
        self.num_cycle = num_cycle

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        muts = self.data[idx]['mutant'].split(':')
        mut_seq = self.target_seq
        for mut in muts:
            assert mut_seq[int(mut[1:-1])-1] == mut[0]
            mut_seq = mut_seq[:int(mut[1:-1])-1] + mut[-1] + mut_seq[int(mut[1:-1]):]
        data = str2inputs(mut_seq, num_cycle=self.num_cycle)
        return data, self.data[idx]['DMS_score']
    
def protein_data_builder(name, num_cycle, stratify=False):
    data = load_dataset('ICML2022/ProteinGym', data_files=f'ProteinGym_substitutions/{name}.csv', split='train')
    if stratify:
        df = pd.DataFrame(data)
        bins = pd.cut(df['DMS_score'], bins=10, labels=False)
        min_val = bins.value_counts().min()
        undersampled_data = pd.DataFrame()
        for bin_val in bins.unique():
            bin_data = df[bins == bin_val]
            undersampled_data = pd.concat([undersampled_data, bin_data.sample(min_val)])
        undersampled_bins = pd.cut(undersampled_data['DMS_score'], bins=8, labels=False)
    else:
        undersampled_data = pd.DataFrame(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in split.split(np.zeros(len(undersampled_data)), undersampled_bins):
        train_data = undersampled_data.iloc[train_index].to_dict('records')
        test_data = undersampled_data.iloc[test_index].to_dict('records')
    print(len(train_data), len(test_data))
    return ProteinData(name, num_cycle, train_data), ProteinData(name, num_cycle, test_data)

def protein_full_data(name, num_cycle):
    data = load_dataset('ICML2022/ProteinGym', data_files=f'ProteinGym_substitutions/{name}.csv', split='train')
    return ProteinData(name, num_cycle, data)