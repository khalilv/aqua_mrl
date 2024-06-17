from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepLabv3:

    def __init__(self, checkpoint_path):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(2)
        self.model.to(self.device)
        self.transforms = A.Compose([
            A.Resize(
                320,
                416,
                always_apply=True,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        return

    def load_model(self, n_classes):
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=True, progress=True)
        # model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT, progress=True)
        
        # freeze weights
        for param in model.parameters():
            param.requires_grad = False
        # replace classifier
        model.classifier = DeepLabHead(960, num_classes=n_classes)
        return model

    def segment(self, img):
        transformed_img = self.transforms(image=img)
        transformed_img = transformed_img['image'].to(self.device)
        transformed_img = torch.unsqueeze(transformed_img, 0)
        outputs = self.model(transformed_img)['out'].squeeze()  # run model on inputs
        return torch.argmax(outputs, dim=0).detach().cpu().numpy() #return mask
