import h5py
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.data.transforms import DataTransform  # DataTransform 임포트

class SliceData(Dataset):
    def __init__(self, root, transform, sample_rate=1.0, challenge="singlecoil", input_key="kspace", target_key="kspace"):
        self.root = root
        self.transform = transform
        self.sample_rate = sample_rate
        self.challenge = challenge
        self.input_key = input_key
        self.target_key = target_key
        self.examples = []
        self._load_files()

        if sample_rate < 1.0:
            self.examples = random.sample(self.examples, int(len(self.examples) * sample_rate))

    def _load_files(self):
        for subdir in ['image', 'kspace']:
            files = list(Path(self.root, subdir).iterdir())
            for fname in sorted(files):
                if fname.is_file():  # 디렉토리가 아닌 파일만 처리
                    num_slices = self._get_metadata(fname)
                    self.examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

    def _get_metadata(self, fname):
        num_slices = 0
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_idx = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf[self.input_key][slice_idx]
            target = hf[self.target_key][slice_idx] if self.challenge == 'multicoil' else kspace
            if kspace is None or target is None:
                print(f"Found None value in kspace or target at file {fname} slice {slice_idx}")
            return self.transform(kspace, target, fname.name, slice_idx)

def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, args.max_key),
        input_key=args.input_key,
        target_key=args.target_key,
        sample_rate=args.sample_rate
    )

    print(f"Total samples: {len(data_storage)}")

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader