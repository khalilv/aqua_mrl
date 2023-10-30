import rclpy
import cv2
import cv_bridge
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

class pipeline_segmentation(Node):

    def __init__(self):
        super().__init__('pipeline_segmentation')
        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/back/image_raw/compressed',
            self.camera_callback,
            10)
        self.cv_bridge = cv_bridge.CvBridge()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(2)
        self.model.to(self.device)
        self.transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        checkpoint = torch.load(
            'src/aqua_pipeline_inspection/pipeline_segmentation/models/deeplabv3/best.pt', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        cv2.namedWindow("Pipeline", cv2.WINDOW_AUTOSIZE)
        print('Initialized: pipeline_segmentation')

    def camera_callback(self, msg):
        t0 = time.time()
        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        transformed_img = self.transforms(image=img)
        transformed_img = transformed_img['image'].to(self.device)
        transformed_img = torch.unsqueeze(transformed_img, 0)
        outputs = self.model(transformed_img)[
            'out'].squeeze()  # run model on inputs        
        pred = np.stack((torch.argmax(outputs, dim=0).detach(
        ).cpu().numpy() * 255,)*3, axis=-1).astype(np.uint8)
        concat = np.concatenate((pred, img), axis=1)
        cv2.imshow('Pipeline', concat)
        cv2.waitKey(1)
        t1 = time.time()
        print('Total processing time: ', (t1 - t0))
        return

    def load_model(self, n_classes):
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True)
        # freeze weights
        for param in model.parameters():
            param.requires_grad = False
        # replace classifier
        model.classifier = DeepLabHead(2048, num_classes=n_classes)
        return model


def main(args=None):
    rclpy.init(args=args)

    subscriber = pipeline_segmentation()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
