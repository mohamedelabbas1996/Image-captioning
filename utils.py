import torchvision


def pad_sequence(seq, maxlen):
    "pads sequence with zeros"
    return [0]*(maxlen - len(seq)) + seq


def preprocess(s):
    s = s.lower()
    for char in "!@#$%^&*()-+?/><';|\\.,`~:":
        s = s.replace(char,"")
    s = s.replace("\s+"," ")
    return 'startseq '+s+' endseq'


def get_transforms():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((299, 299))
    ])
    return transform

