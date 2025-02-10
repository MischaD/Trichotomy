from einops import repeat
from torch.nn.functional import binary_cross_entropy
from src.privacy import SiameseNetwork
import torch
from torchvision.transforms import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as T 
import torch.nn.functional as F
import torchvision
from .utils import class_labels


DEFAULT_CLF_PATH =  "/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/results_chexnet_real/saved_models_cxr8/m-05122024-131940.pth.tar"
DEFAULT_PRIV_PATH = "/vol/ideadata/ed52egek/pycharm/trichotomy/privacy/archive/Siamese_ResNet50_allcxr/Siamese_ResNet50_allcxr_checkpoint.pth"


class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
    
        super(DenseNet121, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
        
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class ToTensorIfNotTensor:
    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            return input
        return T.ToTensor()(input)
    


class Classifier(nn.Module): 
    def __init__(self, model, transforms="default", device="cuda") -> None:
        super().__init__()
        self.device = device
        if transforms == "default": 
            normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            transformList = []
            #transformList.append(T.Resize(256)) -- forward pass during inference uses tencrop 
            transformList.append(T.Resize(226))
            transformList.append(T.CenterCrop(226))
            transformList.append(ToTensorIfNotTensor())
            transformList.append(normalize)
            self.transforms=T.Compose(transformList)
        else: 
            self.transforms = transforms

        self.model = model
        self.model.eval()
        

    def forward(self, x): 
        x_in = self.transforms(x)
        return self.model(x_in)
    
    def lazy_foward(self, x): 
        # accepts tensor, 0-1, bchw 
       
        with torch.no_grad():
            x_in = self.transforms(x)
            if x_in.dim() == 3: 
                x_in = x_in.unsqueeze(dim=0)
            
            varInput = x_in

            features = self.model.densenet121.features(varInput)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            hidden_features = torch.flatten(out, 1)
            out = self.model.densenet121.classifier(hidden_features)
            #outMean = out.view(bs, ).mean(1)
        return out.data, hidden_features.data


def get_classification_model(model_path): 
    cudnn.benchmark = True
    
    #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
    model = DenseNet121(len(class_labels), True)

    modelCheckpoint = torch.load(model_path)
    state_dict = {k[7:]:v for k, v in modelCheckpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return Classifier(model)


def get_privacy_model(path="/vol/ideadata/ed52egek/pycharm/trichotomy/privacy/archive/Siamese_ResNet50_allcxr/Siamese_ResNet50_allcxr_checkpoint.pth"): 
    net = SiameseNetwork()
    net.load_state_dict(torch.load(path)["state_dict"])
    return net


class DiADMSampleEvaluator(): 
    def __init__(self, device, clf_path=DEFAULT_CLF_PATH, priv_path=DEFAULT_PRIV_PATH) -> None:
        self.privnet = get_privacy_model(path=priv_path) if priv_path is not None else get_privacy_model()
        self.privnet = self.privnet.to(device)
        self.privnet.model = self.privnet.model.to(device)

        self.clf_model = get_classification_model(clf_path)
        self.clf_model = self.clf_model.to(device)
        self.clf_model.model = self.clf_model.model.to(device)

    def lazy_predict(self, batch): 
        # 0 - 1, size does not matter
        # batch[0] is real image, 
        # batch[1:] are synthetic images

        pred, f_clf = self.clf_model.lazy_foward(batch)
        clf_pred_scores = binary_cross_entropy(repeat(pred[0], "f -> b f", b=len(pred[1:])), pred[1:], reduction='none')
        clf_pred_scores = clf_pred_scores.mean(dim=1)

        real = repeat(batch[0], "c h w -> b c h w", b=len(batch[1:]))
        snth = batch[1:]

        priv_pred = self.privnet.lazy_pred(real, snth)
        return clf_pred_scores, priv_pred.squeeze()

    def predict(self, real_image, snth_image_batch):
        return self.lazy_predict(torch.cat(real_image, snth_image_batch))