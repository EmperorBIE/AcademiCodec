# 和 Encodec* 的 dataset.py 有点类似但是不完全一样
# 主要是 prob > 0.7 的时候多了 ans2
import glob
import json
import random

import torch
import torchaudio
from torch.utils.data import Dataset

import logging

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = []
        logging.info(f"Extracting datasets from path '{audio_dir}'")
        # self.filenames.extend(glob.glob(audio_dir + "/*.wav"))
        self.filenames.extend(glob.glob(audio_dir + "/*.flac")) # For LibriSpeech, should use flac as suffix
        logging.info(f"Extract finished, file list size: {len(self.filenames)}")
        _, self.sr = torchaudio.load(self.filenames[0])
        logging.info(f"Sampling rate: {self.sr}")
        self.max_len = 48000  # 48000, 3sec

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #print(self.filenames[index])
        prob = random.random()  # (0,1)
        if prob > 0.7:
            # data augmentation
            ans1 = torch.zeros(1, self.max_len)
            ans2 = torch.zeros(1, self.max_len)
            audio1 = torchaudio.load(self.filenames[index])[0]
            index2 = random.randint(0, len(self.filenames) - 1)
            audio2 = torchaudio.load(self.filenames[index2])[0]
            if audio1.shape[1] > self.max_len:
                st = random.randint(0, audio1.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans1 = audio1[:, st:ed]
            else:
                ans1[:, :audio1.shape[1]] = audio1
            if audio2.shape[1] > self.max_len:
                st = random.randint(0, audio2.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans2 = audio2[:, st:ed]
            else:
                ans2[:, :audio2.shape[1]] = audio2
            ans = ans1 + ans2
            return ans
        else:
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
    """Dataset to load NSynth data from a JSON file."""

    def __init__(self, json_dir, group_id=[]):
        super().__init__()
        self.max_len = 24000  # 24000, 1.5sec
        self.filenames = []
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
        #print(self.filenames[index])
        prob = random.random()  # (0,1)
        if prob > 0.7:
            # data augmentation
            ans1 = torch.zeros(1, self.max_len)
            ans2 = torch.zeros(1, self.max_len)
            audio1 = torchaudio.load(self.filenames[index])[0]
            index2 = random.randint(0, len(self.filenames) - 1)
            audio2 = torchaudio.load(self.filenames[index2])[0]
            if audio1.shape[1] > self.max_len:
                st = random.randint(0, audio1.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans1 = audio1[:, st:ed]
            else:
                ans1[:, :audio1.shape[1]] = audio1
            if audio2.shape[1] > self.max_len:
                st = random.randint(0, audio2.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans2 = audio2[:, st:ed]
            else:
                ans2[:, :audio2.shape[1]] = audio2
            ans = ans1 + ans2
            return ans
        else:
            ans = torch.zeros(1, self.max_len)
            audio = torchaudio.load(self.filenames[index])[0]
            if audio.shape[1] > self.max_len:
                st = random.randint(0, audio.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                return audio[:, st:ed]
            else:
                ans[:, :audio.shape[1]] = audio
                return ans
