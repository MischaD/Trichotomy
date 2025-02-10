import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.transforms import functional as F


class ToTensorIfNotTensor:
    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            return input
        return F.to_tensor(input)


class SiameseNetwork(nn.Module):
    def __init__(self, network='ResNet-50', in_channels=3, n_features=128, device="cuda"):
        super(SiameseNetwork, self).__init__()
        self.network = network
        self.in_channels = in_channels
        self.n_features = n_features

        if self.network == 'ResNet-50':
            # Model: Use ResNet-50 architecture
            self.model = models.resnet50(pretrained=True)
            # Adjust the input layer: either 1 or 3 input channels
            if self.in_channels == 1:
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif self.in_channels == 3:
                pass
            else:
                raise Exception(
                    'Invalid argument: ' + self.in_channels + '\nChoose either in_channels=1 or in_channels=3')
            # Adjust the ResNet classification layer to produce feature vectors of a specific size
            self.model.fc = nn.Linear(in_features=2048, out_features=self.n_features, bias=True)

        else:
            raise Exception('Invalid argument: ' + self.network +
                            '\nChoose ResNet-50! Other architectures are not yet implemented in this framework.')

        self.fc_end = nn.Linear(self.n_features, 1)
        self.transform_lazy = None
        if device is not None: 
            self.model = self.model.to("cuda")

    def forward_once(self, x):

        # Forward function for one branch to get the n_features-dim feature vector before merging
        output = self.model(x)
        output = torch.sigmoid(output)
        return output

    def forward(self, input1, input2):

        # Forward
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)

        return output


    def setup_transforms(self, image_size): 
        self.transform_lazy = Compose([
                    Resize((image_size, image_size)),
                    ToTensorIfNotTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def lazy_forward(self, input1, input2, image_size=256): 
        if self.transform_lazy is None: 
            self.setup_transforms(image_size=image_size)

        input1 = self.transform_lazy(input1)
        input2 = self.transform_lazy(input2)

        if input1.dim() == 3: 
            input1 = input1.unsqueeze(dim=0)

        if input2.dim() == 3:
            input2 = input2.unsqueeze(dim=0)

        with torch.no_grad(): 
            out = self.forward(input1, input2)

        return out


    def lazy_forward_once(self, input1, image_size=256): 
        if self.transform_lazy is None: 
            self.setup_transforms(image_size=image_size)

        input1 = self.transform_lazy(input1)
        if input1.dim() == 3: 
            input1 = input1.unsqueeze(dim=0)

        with torch.no_grad(): 
            out = self.forward_once(input1)
        return out


    def feat_to_pred(self, feat1, feat2):
        difference = torch.abs(feat1 - feat2)
        with torch.no_grad():
            output = self.fc_end(difference)
        out = torch.sigmoid(output)
        out = (out > 0.5).int().item()
        return out 


    def lazy_pred(self, input1, input2, image_size=256): 
        output = self.lazy_forward(input1, input2, image_size)
        out = torch.sigmoid(output)
        out = (out > 0.5).int()
        return out 