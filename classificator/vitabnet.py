import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

class ViTabNet(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ViTabNet, self).__init__()
        self.tabular_fc1 = nn.Linear(9, 32) ## add re
        self.tabular_fc2 = nn.Linear(32, 64)
        self.tabular_fc3 = nn.Linear(64, 128)

        self.visual_model = models.resnet18(pretrained=True)

        num_ftrs = self.visual_model.fc.in_features
        self.visual_model.fc = nn.Linear(num_ftrs, 128)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=2)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x_visual, x_tabular):
        # Process tabular data
        x_tabular = self.tabular_fc1(x_tabular)
        x_tabular = nn.ReLU()(x_tabular)
        x_tabular = self.tabular_fc2(x_tabular)
        x_tabular = nn.ReLU()(x_tabular)
        x_tabular = self.tabular_fc3(x_tabular)
        x_tabular = nn.ReLU()(x_tabular)

        # Process visual data
        x_visual = self.visual_model(x_visual)
        x_visual = nn.ReLU()(x_visual)

        # Combine tabular and visual data and apply transformer encoding
        x = torch.cat((x_tabular, x_visual), dim=1)
        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)

        # Apply fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Example usage
model = ViTabNet(num_classes=10).cuda()
x_image = torch.ones([1, 3, 224,224])
x_table = torch.ones([1, 9])
summary(model, input_data=(x_image, x_table))

output = model(x_image.cuda(), x_table.cuda())
print(output)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
#
# # Load data and train model using DataLoader
# train_dataset = MyDataset(...)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# for epoch in range(num_epochs):
#     for x_tabular, x_visual, y in train_loader:
#         optimizer.zero_grad()
#         y_pred = model(x_tabular, x_visual)
#         loss = criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
