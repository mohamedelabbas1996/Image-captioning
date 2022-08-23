import torch.nn

from model import ImageCaptionNetwork
from torch.optim import Adam
from dataset import ImageCaptionsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
def train(dataloader , optimizer, model, criterion , n_epochs):
    for epoch in (range(n_epochs)):
        for img_feature, caption_sequence , output_token in tqdm(dataloader):
            #print (img_feature.shape,caption_sequence.shape,output_token.shape)
            output = model(img_feature,caption_sequence)
            print(output_token.shape)
            optimizer.zero_grad()
            loss = criterion(output,torch.tensor(output_token))
            print (loss.item())
            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    dataset = ImageCaptionsDataset("data/Flicker8k_Dataset", "data/Flickr8k_text/Flickr8k.token.txt")
    loader = DataLoader(dataset, batch_size= 16, shuffle= True)
    loss = torch.nn.CrossEntropyLoss()
    vocab_size = 8918
    model = ImageCaptionNetwork(vocab_size)
    optimizer = Adam(params= model.parameters())

    train(loader, optimizer,model,loss,1)


