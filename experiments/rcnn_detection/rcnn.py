import torch
import lightning as L
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class FasterRCNNLit(L.LightningModule):
    def __init__(self, backbone, num_classes, lr, image_size=256):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=image_size,
            max_size=image_size
        )

        self.lr = lr

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images = batch["images"]  # [B, 1, H, W]
        targets = batch["targets"]  # List[Dict]
        orig_size = batch["orig_size"]  # [B, 2]

        images = [img.to(self.device) for img in images]

        # xywhn to xyxy conversion
        for i, target in enumerate(targets):
            img_height, img_width = orig_size[i]
            target['boxes'] = self._xywhn_to_xyxy(
                target['boxes'],
                img_width,
                img_height
            )

        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss
    
    def _xywhn_to_xyxy(self, boxes, img_width, img_height):
        xyxy = boxes.clone()
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_width   # x_min
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_height  # y_min
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_width   # x_max
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_height  # y_max
        return xyxy

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
