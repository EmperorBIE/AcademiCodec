import glob
import random

import torch
import torchaudio
from torch.utils.data import Dataset

import json

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir + "/*.wav"))
        print(len(self.filenames))
        _, self.sr = torchaudio.load(self.filenames[0])
        self.max_len = 24000  # 24000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            return audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio
            return ans

class JsonDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, json_dir, group_id=[]):
        super().__init__()
        self.filenames = []
        self.max_len = 24000  # 24000

        with open(json_dir, 'r') as f:
            self.data_info = json.load(f)

        if len(group_id) == 0:
            self.filenames = [item['file_path'] for item in self.data_info.values()]
        else:
            for item in self.data_info.values():
                if item['groupID'] in group_id:
                    self.filenames.append(item['file_path'])
        
        if self.filenames:
            _, self.sr = torchaudio.load(self.filenames[0])
        else:
            self.sr = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            return audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio
            return ans