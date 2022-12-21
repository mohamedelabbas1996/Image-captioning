import torch
import pandas as pd
from utils import pad_sequence, preprocess, get_transforms
import PIL
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, word_map, max_len=46, captions_file='/kaggle/input/flickr8k/captions.txt',
                 image_dir='/kaggle/input/flickr8k/Images'):
        super().__init__()
        self.image_dir = image_dir
        self.max_len = max_len
        self.df = pd.read_csv(captions_file)
        self.df['caption'] = self.df['caption'].apply(preprocess)
        self.word_mapping = word_map

    def captions_vocab_size(self):
        return self.word_mapping.vocab_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        transform = get_transforms()
        image_file_name, caption = self.df['image'][index], self.df['caption'][index]
        image = PIL.Image.open(os.path.join(self.image_dir, image_file_name))
        image_tensor = transform(image)
        tokenized_caption = self.word_mapping.tokenize_string(caption)
        padded_caption = pad_sequence(tokenized_caption, self.max_len)
        target = pad_sequence(tokenized_caption[1:], self.max_len)
        return image_tensor, torch.tensor(padded_caption), torch.tensor(target)

