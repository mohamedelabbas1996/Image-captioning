import torch
import torchvision
from PIL import Image
import pickle
import tqdm
import os
from collections import defaultdict
from model import get_vgg16
from tokenizer import Tokenizer


def pad_sequence(seq, maxlen):
    return [0]*(maxlen- len(seq)) + seq


def extract_features(directory, model,load_from_file=None, save_to_file=None, n_images=300):
    features = dict()
    if load_from_file:
        with open(load_from_file,"rb") as file:
            features = pickle.load(file)
        return features
    if not n_images:
        n_images = len(os.listdir(directory))
    for image in tqdm.tqdm(os.listdir(directory)[:n_images]):
        img_id = image.split(".")[0]
        features[img_id] = extract_image_features(directory + "/" + image, model)
    if save_to_file:
        pickle.dump(features, open(save_to_file, "wb"))
    return features


def extract_image_features(image_file, model):
    image = Image.open(image_file)
    image_tensor = torchvision.transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    return model(image_tensor)


def pre_process_caption(caption:str):
    # convert to lower case
    caption = caption.lower()
    caption = caption.replace("[^A-Za-z]", "")
    caption = caption.replace("\s+", " ")
    caption = " ".join([word for word in caption.split() if len(word)>1 ])
    caption = f"startseq  {caption} endseq"
    return caption


def create_id_caption_mapping(captions_file):
    mapping = defaultdict(list)
    captions = list()
    with open(captions_file, "r") as f:
        for line in f.readlines():
            img_id = line.split()[0]
            # remove the caption number
            img_id = img_id.split("#")[0]
            # remove image extension
            img_id = img_id.split(".")[0]
            img_caption = line.split()[1:]
            caption = " ".join(img_caption)
            captions.append(caption)
            mapping[img_id].append(pre_process_caption(caption))
    return mapping, captions


def prepare_sequence_teacher_forcing(id_caption_mapping, features, tokenizer,maxlength, vocab_size):
    data = []
    for img_id, captions in id_caption_mapping.items():
        if img_id not in features:
            continue
        for caption in captions:
            seq = tokenizer.text_to_sequence(caption)
            for idx, token in enumerate(seq):
                padded_seq = torch.tensor(pad_sequence(seq[:idx], maxlen=maxlength))
                one_hot_output = torch.tensor(torch.nn.functional.one_hot(torch.tensor([token]), num_classes = vocab_size).tolist()[0])
                data.append([features[img_id],padded_seq, one_hot_output])
    return data


if __name__ == "__main__":

    mapping , captions = create_id_caption_mapping("data/Flickr8k_text/Flickr8k.token.txt")
    print (max([len(caption) for caption in captions]))
    #     print (img_id, caption)
    #     break
    #extract_features("data/Flicker8k_Dataset",get_vgg16())
    # print (pad_sequence([1,2,3],2))
    # features = {"abcd": [1,2,3]}
    # mapping = {"abcd" : ["the girl is fat"]}
    # tokenizer = Tokenizer()
    # tokenizer.fit_text(mapping["abcd"])
    # for sample in prepare_sequence_teacher_forcing(mapping,features,tokenizer,0,tokenizer.vocab_size):
    #     print (sample)

