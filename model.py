import torchvision
import torch
def get_vgg16():
    vgg16 = torchvision.models.vgg16()
    vgg16.classifier[6] = torch.nn.Identity()
    return vgg16
