import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix

labels = ['Biogenic_Organic', 'Metallic', 'Mineral', 'Tire wear']
model_path = '/media/lecun/HD/Expor2/Particle-classifier/classification/models/export2_model.pth'

def main():
    TRAIN_DATA_PATH = "/media/lecun/HD/Expor2/ParticlesDB/folders/val/"

    test_data = ImageFolder(root=TRAIN_DATA_PATH,
                            transform=get_preprocessing())
    test_loader = DataLoader(test_data, batch_size=32,
                             shuffle=True, num_workers=4)

    one_batch = next(iter(test_loader))
    model = InferenceModel(model_path, labels)
    true_labels, pred_probs = test(test_loader, model)
    pred_labels = np.argmax(pred_probs, axis=-1)

    # Prints classification report on test data
    report = classification_report(true_labels, pred_labels,
                                   target_names=labels)
    print('Classification Report')
    print(report)

    # Confusion matrix
    cnf_matrix = confusion_matrix(true_labels, pred_labels)
    # Plot non-normalized confusion matrix
    print(cnf_matrix)

def get_preprocessing(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])

def test(test_loader, model):

    targets = torch.autograd.Variable().cuda()
    predictions = torch.autograd.Variable().cuda()

    for i, (input, target) in enumerate(test_loader):
        if i > int(len(test_loader)):
            break

        input, target = input.cuda(), target.float().cuda()
        output = model(input)

        targets = torch.cat((targets, target), 0)
        predictions = torch.cat((predictions, output.float()), 0)

    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    return targets, predictions

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
    main()






