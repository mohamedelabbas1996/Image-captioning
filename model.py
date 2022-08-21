import torchvision
import torch


def get_vgg16():
    vgg16 = torchvision.models.vgg16()
    vgg16.classifier[6] = torch.nn.Identity()
    return vgg16


class Encoder(torch.nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.img_features = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256),
            torch.nn.ReLU()
        )
        self.sequence = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size),
            torch.nn.Dropout(0.4),
            torch.nn.LSTM(256)

        )

    def forward(self, img_feature, caption_sequence):
        return torch.add(self.img_features(img_feature), self.sequence(caption_sequence))


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(256),
                torch.nn.ReLU(),
                torch.nn.Linear(vocab_size),
                torch.nn.Softmax()
            )

    def forward(self, encoded_features):
        return self.layers(encoded_features)


class EncoderDecoder(torch.nn.Module):
    def __int__(self,encoder, decoder):
        super(EncoderDecoder, self).__int__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img_feature, caption_sequence):
        encoder_output = self.encoder(img_feature, caption_sequence)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
