from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        return len()
    
    def __getitem__(self, index: int):
        return 