import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class MloDataset(Dataset):
    def __init__(self, input_data):
        """
        TrainDataset Initializer
        
        Attributes:
        input_data -- Input data with email embeddings, features, and labels
        """
        self.n_feature = len(input_data)
        self.data = input_data
        
    def __len__(self):
        return len(self.data[0])
        
    def __getitem__(self, item):
        ret = [d[item] for d in self.data]
        return ret

def create_dataset(data=tuple):
    def parse_data(data, target_type=str):
        targets = []
        for d in data:
            dl = len(d) // 3
            if target_type == 'train':
                targets.extend(d[:dl])
            elif target_type == 'valid':
                targets.extend(d[dl:(2*dl)])
            elif target_type == 'test':
                targets.extend(d[(2*dl):(3*dl)])
            else:
                assert(0)

        return targets

       

    train_data = parse_data(data, target_type='train')
    valid_data = parse_data(data, target_type='valid')
    test_data = parse_data(data, target_type='test')

    train_dataset = MloDataset(train_data)
    valid_dataset = MloDataset(valid_data)
    test_dataset = MloDataset(test_data)

    return train_dataset, valid_dataset, test_dataset

def get_dataloader(dataset, shuffle, param_dict):
    def data_collate_fn(samples):
        data = tuple(zip(*samples))
        agg_data = tuple([pad_sequence(d, padding_value=0, batch_first=True) for d in data[:-1]])
        label = torch.Tensor(data[-1]).float()

        return agg_data, label

    dataloader = DataLoader(
        dataset,
        batch_size=param_dict['training_param']['batch_size'],
        num_workers=param_dict['training_param']['num_workers'],
        shuffle=shuffle,
        collate_fn=data_collate_fn
    )

    return dataloader
 
