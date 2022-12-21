import torch
import torchvision


class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.cnn = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, embed_size)
        for name, param in self.cnn.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.cnn(x)


# create model

class ImageCaption(torch.nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.cnn = EncoderCNN(embed_size)
        self.batchnorm = torch.nn.BatchNorm1d(num_features=embed_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.embed = torch.nn.Embedding(embedding_dim=embed_size, num_embeddings=vocab_size)
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.relu = torch.nn.ReLU()

    def forward(self, image, caption):
        # print (image.shape)
        # image embedding
        img_features = self.dropout(image)
        img_features = self.cnn(img_features)

        img_features = img_features.unsqueeze(1)
        # print ("img_features ", img_features.shape)
        # text embedding
        embed = self.embed(caption)
        embed = self.dropout(embed)
        # print ("text embedding ", embed.shape)

        # image encoding using lstm
        _, h = self.lstm(img_features)
        # print ('img_features embedding', h[0].shape, h[1].shape)

        output, _ = self.lstm(embed, h)
        # print ("lstm output shape " ,output.shape)

        # passing output through linear layer to convert to shape
        output = self.linear(output)
        output = output.view(-1, self.vocab_size)

        return output

    def generate_caption(self, image_file, word_map):
        pass
