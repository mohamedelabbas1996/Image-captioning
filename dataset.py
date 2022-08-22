from torch.utils.data import Dataset
from preprocessing import create_id_caption_mapping, extract_features, prepare_sequence_teacher_forcing
from tokenizer import Tokenizer


class ImageCaptionsDataset(Dataset):
    def __init__(self, images_path, captions_path):
        self.features = extract_features(images_path)
        self.img_captions_mapping, self.all_captions = create_id_caption_mapping(captions_path)
        tokenizer = Tokenizer()
        tokenizer.fit_text(self.all_captions)
        self.data = prepare_sequence_teacher_forcing(
            self.img_captions_mapping,
            self.features, tokenizer, maxlength=1,  vocab_size=tokenizer.vocab_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
