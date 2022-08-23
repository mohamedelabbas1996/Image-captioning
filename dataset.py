from torch.utils.data import Dataset
from preprocessing import create_id_caption_mapping, extract_features, prepare_sequence_teacher_forcing
from tokenizer import Tokenizer
from model import get_vgg16
from torch.utils.data import DataLoader

class ImageCaptionsDataset(Dataset):
    def __init__(self, images_path, captions_path):
        self.features = extract_features(images_path,model = get_vgg16() ,load_from_file="features.pkl")
        self.img_captions_mapping, self.all_captions = create_id_caption_mapping(captions_path)
        tokenizer = Tokenizer()
        tokenizer.fit_text(self.all_captions)
        print ("vocab_size", tokenizer.vocab_size)
        self.data = prepare_sequence_teacher_forcing(
            self.img_captions_mapping,
            self.features, tokenizer, maxlength=199,  vocab_size=tokenizer.vocab_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    dataset = ImageCaptionsDataset("data/Flicker8k_Dataset","data/Flickr8k_text/Flickr8k.token.txt")
    #dataset[0][0].shape
    loader = DataLoader(dataset,batch_size= 16, shuffle=True)
    for t in loader:
        print((t[0].shape),t[1].shape)
        break