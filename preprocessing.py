import torch
import torchvision
from PIL import Image
import pickle
import tqdm
import os
from collections import defaultdict
from tokenizer import Tokenizer


def pad_sequence(seq, maxlen):
    return [0]*(maxlen- len(seq)) + seq


def extract_features(directory, model, save=True):
    features = dict()
    for image in tqdm.tqdm(os.listdir(directory)):
        features[image] = extract_image_features(directory + "/" + image, model)
    if save:
        pickle.dump(features, open("features.pkl", "wb"))
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


def create_id_caption_mapping(file):
    mapping = defaultdict(list)
    captions = list()
    with open(file, "r") as f:
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
        for caption in captions:
            seq = tokenizer.text_to_sequence(caption)
            for idx, token in enumerate(seq):
                padded_seq = pad_sequence(seq[:idx], maxlen=maxlength)
                one_hot_output = torch.nn.functional.one_hot(torch.tensor([token]), num_classes = vocab_size+1).tolist()[0]
                data.append([features[img_id],padded_seq, one_hot_output])
    return data


if __name__ == "__main__":
    # for img_id,caption in create_id_caption_mapping("data/Flickr8k_text/Flickr8k.token.txt").items():
    #     print (img_id, caption)
    #     break
    print (pad_sequence([1,2,3],2))
    features = {"abcd": [1,2,3]}
    mapping = {"abcd" : ["the girl is fat"]}
    tokenizer = Tokenizer()
    tokenizer.fit_text(mapping["abcd"])
    for sample in prepare_sequence_teacher_forcing(mapping,features,tokenizer,0,tokenizer.vocab_size):
        print (sample)

