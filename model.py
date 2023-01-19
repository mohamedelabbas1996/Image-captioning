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

    def generate_caption(self, image_file, word_map, maxlen=46):
        """
        Generates a caption for a given image file using image features extracted from the Inception net, and a word map.

        Parameters:
            - image_file (str): path to the image file
            - word_map (WordMapping): an object that contains the mapping between words and their corresponding token IDs
            - maxlen (int): maximum length of the generated caption

        Returns:
            - str: the generated caption
        """
        # Open image file
        image = PIL.Image.open(image_file)

        # Apply image transforms
        transforms = get_transforms()
        image = transforms(image)
        image = image.unsqueeze(0)

        # Extract image features using Inception net
        img_features = self.cnn(image)

        # Generate hidden and cell state by passing image features to the LSTM
        _, h = self.lstm(img_features)

        # Initialize list to store generated caption
        generated_caption = []

        # Start with the "startseq" token
        caption = "startseq"

        # Tokenize the caption
        tokenized_caption = torch.tensor(word_map.tokenize_string(caption))

        # Iterate for the maximum length of the caption
        for _ in range(maxlen):
            # Embed the tokenized caption
            embed = self.embed(tokenized_caption)

            # Pass the embedded caption through the LSTM
            tokenized_caption, h = self.lstm(embed, h)

            # Get the index of the maximum value in the output
            tokenized_caption = tokenized_caption.argmax(dim=1)

            # If the token corresponds to "endseq", end the caption generation
            if word_map.token_to_word(tokenized_caption.item()) == 'endseq':
                break

            # Add the token to the generated caption
            generated_caption.append(tokenized_caption.item())

        # Convert the generated caption to text using the word map
        return word_map.sequence_to_text(generated_caption)
