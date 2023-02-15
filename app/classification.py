import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

labels = ['Biogenic_Organic', 'Metallic', 'Mineral', 'Tire wear']
model_path = './export2_model.pth'

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
    """
    Class for hosting a ResNet-18 model and performing inference on images.

    Args:
        model_path (str): The path to the saved PyTorch model checkpoint file.
            If `None`, an untrained model will be used.
        labels (List[str]): A list of string labels for the model's output classes.

    Returns:
        output (torch.Tensor): The softmax probabilities of the input image belonging
            to each of the output classes.

    """

    def __init__(self, model_path = None, labels= None):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device

        self.classifier = ResNet18(len(labels))

        if model_path is not None:
            checkpoint = torch.load(model_path , map_location=device)
            state_dict = checkpoint['model']
            self.classifier.model.load_state_dict(state_dict)

        self.classifier.eval()
        self.classifier.to(self.device)

    def __call__(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.classifier(input_tensor)
        return output

class ResNet18(nn.Module):
    """
     A modified ResNet-18 model for image classification.

    Args:
        out_size (int): The number of classes in the classification task.
        include_top (bool): Whether to include the fully-connected classification
            layer on top of the ResNet-18 backbone. If set to False, the model
            will output features directly.

    Returns:
        output (torch.Tensor): The softmax probabilities of the input image belonging
            to each of the output classes.

    """

    def __init__(self, out_size, include_top=True):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features

        if include_top:
            self.model.fc = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        # Modified for return output and feature map for CAM's
        out = self.model(x.float())
        output = nn.functional.softmax(out, -1)
        return output

if __name__ == "__main__":
    import os
    import cv2 as cv

    # TRAIN_DATA_PATH = "/media/lecun/HD/Expor2/ParticlesDB/folders/val/Tire wear"
    # img_path = os.listdir(TRAIN_DATA_PATH)[0]
    # img = cv.imread(os.path.join(TRAIN_DATA_PATH, img_path))

    # img = np.zeros((512,512,3)).astype(np.uint8)
    # model = InferenceModel(model_path, labels)
    #
    # preds = clasiffy_img(img, model)
    # print(dict(zip(labels,preds[0])))






