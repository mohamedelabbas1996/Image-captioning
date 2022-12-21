import argparse
from dataset import Dataset
from torch.utils.data import DataLoader
from model import ImageCaption
from train import train
from word_mapping import WordMapping
from utils import get_transforms
import torch
import PIL
import matplotlib.pyplot as plt
from PIL import Image

IMAGES_PATH = "data/Flicker8k_Dataset"
CAPTIONS_PATH = "data/Flickr8k_text/captions.txt"


def main(mode='test', checkpoint=None, learning_rate=0.001, batch_size=16, n_epochs=1, img_file=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == "test":
        if not img_file:
            raise Exception("Image file path not provided, "
                            "please provide the path for the image for which you want to generate captions")
        if not checkpoint:
            raise Exception("Checkpoint Not found , please provide a valid check point file")
        checkpoint = torch.load(checkpoint)
        word_map = WordMapping(all_captions_path=CAPTIONS_PATH)
        vocab_size = word_map.vocab_size
        model = ImageCaption(embed_size=512, vocab_size=vocab_size, hidden_size=512, num_layers=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        transforms = get_transforms()
        image = PIL.Image.open(img_file)
        image = transforms(image)
        caption = model.generate_captions(image, word_map)
        print(caption)
        plt.imshow(image)
    elif mode == "train":
        loss = 0
        word_map = WordMapping(all_captions_path=CAPTIONS_PATH)
        vocab_size = word_map.vocab_size
        model = ImageCaption(embed_size=512, vocab_size=vocab_size, hidden_size=512, num_layers=1)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
        dataset = Dataset(word_map=word_map, captions_file=CAPTIONS_PATH, image_dir=IMAGES_PATH)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        if checkpoint:
            checkpoint = torch.load(checkpoint)
            loss = checkpoint['loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        train(dataloader=dataloader, optim=optimizer,
              n_epochs=n_epochs, model=model, device=device, criterion=criterion, loss=loss)
    else:
        raise Exception("Invalid mode")


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('mode', type=str, default='test', help="operation mode can be test or train")
    arg_parser.add_argument('--image_file', type=str, required=False, help="path to image file  to generate caption for")
    arg_parser.add_argument('--checkpoint', type=str, required=False, help="path for model checkpoint")
    arg_parser.add_argument('--n-epochs', type=int, default=1, help="number of epochs for training")
    arg_parser.add_argument('--learning-rate', type=float, default=0.01, help="optimizer learning rate")
    arg_parser.add_argument('--batch-size', type=int, default=16, help="Training batch size")
    arguments = arg_parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode)
