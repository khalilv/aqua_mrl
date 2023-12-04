import torch
import numpy as np
from aqua_rl.YOLOv7.utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging, letterbox
from aqua_rl.YOLOv7.utils.plots import plot_one_box
from aqua_rl.YOLOv7.utils.torch_utils import time_synchronized, TracedModel
from aqua_rl.YOLOv7.models.experimental import attempt_load

class YoloV7:
    def __init__(self, cfg):

        # Initialize parameters
        set_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = cfg.half # half precision only supported on CUDA
        self.confidence_threshold = cfg.confidence_threshold
        self.iou_threshold = cfg.iou_threshold
        # Load model
        self.model = attempt_load(cfg.weights, map_location=self.device)  # create
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(cfg.image_size, s=self.stride)  # check img_size
        if cfg.trace:
            self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.image_size
        self.old_img_b = 1
        self.verbose = cfg.verbose
        return

    def detect(self, img):
        original = img
        img = letterbox(img, self.image_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold, classes=None)
        t3 = time_synchronized()
        outputs = []
        for i, det in enumerate(pred):  # detections per image
            s = ''
            if len(det):
                # Rescale boxes from img_size to original size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if(conf > self.confidence_threshold):
                        x1, y1,x2, y2 = (torch.FloatTensor(xyxy)).detach().cpu().numpy()
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, original, label=label, color=self.colors[int(cls)], line_thickness=1)
                        outputs.append([int(x1),int(y1),int(x2),int(y2)])
            if self.verbose:
                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        return outputs, original
            