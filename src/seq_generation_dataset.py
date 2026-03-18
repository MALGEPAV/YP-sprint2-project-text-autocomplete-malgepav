from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SeqGenerationDataset(Dataset):
    def __init__(self, beginings, endings):
        super().__init__()
        self.beginings = beginings
        self.endings = endings

    def __len__(self):
        return len(self.endings)
    
    def __getitem__(self, index):
        return self.beginings[index], self.endings[index]
    
def create_seq_gen_dataset_from_csv(path_to_csv):
    seq_df = pd.read_csv(path_to_csv)
    seq_df.dropna(inplace=True)

    beginings = seq_df["begining"].to_list()
    endings = seq_df["ending"].to_list()

    return SeqGenerationDataset(beginings=beginings, endings=endings)


def seq_gen_collate_fn(batch):
    beginings, endings =  zip(*batch)
    return list(beginings), list(endings)



