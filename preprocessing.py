import torch
import torchvision
from PIL import Image
import pickle
import tqdm
from collections import defaultdict


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


def create_id_caption_mapping(file):
    mapping = defaultdict(list)
    with open(file, "r") as f:
        for line in f.readlines():
            img_id = line.split()[0]
            # remove the caption number
            img_id = img_id.split("#")[0]
            # remove image extension
            img_id = img_id.split(".")[0]
            img_caption = line.split()[1:]
            mapping[img_id].append(" ".join(img_caption))
    return mapping


if __name__ == "__main__":
    for img_id,caption in create_id_caption_mapping("data/Flickr8k_text/Flickr8k.token.txt").items():
        print (img_id, caption)
        break
