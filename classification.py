import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

labels = ['Biogenic_Organic', 'Metallic', 'Mineral', 'Tire wear']
model_path = '/media/lecun/HD/Expor2/Particle-classifier/classificator/models/export2_model.pth'

def clasiffy_img(img, model):
    preprocesing = get_preprocessing()
    _input = preprocesing(img).unsqueeze(dim=0)
    pred = model(_input).cpu().numpy()
    return pred

def get_preprocessing(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])

class InferenceModel():
    """ Class for hosting model and perform infenrece on images"""

    def __init__(self, model_path = None, labels= None):
        self.classifier = ResNet18(len(labels))

        if model_path is not None:
            checkpoint = torch.load(model_path)  # , map_location=device)
            state_dict = checkpoint['model']
            self.classifier.model.load_state_dict(state_dict)

        self.classifier.eval()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.classifier.to(self.device)

    def __call__(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.classifier(input_tensor)
        return output

class ResNet18(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, include_top=True):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features

        if include_top:
            self.model.fc = nn.Linear(num_ftrs, out_size)

            # self.resnet18.fc = nn.Sequential(
            #     nn.Linear(num_ftrs, out_size),
            #     nn.Softmax(dim=-1)
            #     # nn.Sigmoid()
            # )

    def forward(self, x):
        # Modified for return output and feature map for CAM's
        out = self.model(x.float())
        output = nn.functional.softmax(out, -1)
        return output

if __name__ == "__main__":
    import os
    import cv2 as cv

    TRAIN_DATA_PATH = "/media/lecun/HD/Expor2/ParticlesDB/folders/val/Tire wear"
    list_im = os.listdir(TRAIN_DATA_PATH)

    img = cv.imread(os.path.join(TRAIN_DATA_PATH, list_im[0]))
    model = InferenceModel(model_path, labels)

    preds = clasiffy_img(img, model)
    print(dict(zip(labels,preds[0])))






