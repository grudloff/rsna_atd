import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Resize
import pytorch_lightning as pl
import rsna_atd.config as config
import monai

def split_targets(targets: torch.tensor) -> tuple[torch.tensor]:
    """ Split target into the different tasks. """
    #         (bowel, fluid, kidney, liver, spleen, injured)
    targets = (targets[:, 0:2], targets[:, 2:4], targets[:, 4:7], targets[:, 7:10],
                targets[:, 10:13], targets[:, 13:14])
    return targets

class CT_BaseModel(pl.LightningModule):
    """ Base model for CT scans. This model represents the null prediction, i.e. the model that always predicts
    constant probabilities for each class. This model implements all the training logic, i.e. the loss function,
    the accuracy function and the inference function. Also, it implements the training and validation steps, together
    with the configuration of the optimizer and learning rate scheduler.
    """
    def __init__(self, loss_weights=None, lr=1e-4):
        super().__init__()

        if loss_weights is None:
            loss_weights = config.loss_weights
        self.loss_functions = nn.ModuleDict({})
        for head_name in config.tasks:
            weight = torch.Tensor(list(loss_weights[head_name].values()))
            self.loss_functions[head_name] = nn.CrossEntropyLoss(weight=weight)

        self.lr = lr

    def forward(x):
        B = len(x)
        dtype = x.dtype
        device = x.device
        return [torch.ones((B, i), dtype=dtype, device=device)/i for i in [2, 2, 3, 3, 3, 1]]
    
    def _loss_fn(self, outputs, targets):
        loss = 0
        for i, head_name in enumerate(config.tasks):
            loss_function = self.loss_functions[head_name]
            current_loss = loss_function(outputs[i], targets[i])
            loss += current_loss
            self.log(f'{head_name}_loss', current_loss, on_step=False, on_epoch=True)
        return loss

    def _accuracy_fn(self, output, target):
        batch_size, n_classes = target.shape
        threshold = 1/n_classes
        product = output*target
        accuracy = torch.sum(product > threshold).item()
        accuracy /= batch_size
        return accuracy
    
    def _mean_accuracy_fn(self, outputs, targets, step="train"):
        # Returns the mean accuracy of the batch
        n_tasks = len(targets)
        total = 0
        for i, head_name in enumerate(config.tasks):
            prediction = outputs[i]
            target = targets[i]
            accuracy = self._accuracy_fn(prediction, target)
            total += accuracy
            self.log(f'{step}_{head_name}_accuracy', accuracy, on_step=False, on_epoch=True)
        meanm_accuracy = total/n_tasks
        return meanm_accuracy
    
    def _inference_fn(self, batch):
        images, targets = batch
        outputs = self(images)
        targets = split_targets(targets)
        return outputs, targets

    def training_step(self, batch, batch_idx):
        outputs, targets = self._inference_fn(batch)
        loss = self._loss_fn(outputs, targets)
        mean_accuracy = self._mean_accuracy_fn(outputs, targets)
        self.log(f'train_mean_accuracy', mean_accuracy, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        outputs, targets = self._inference_fn(batch)
        loss = self._loss_fn(outputs, targets)
        mean_accuracy = self._mean_accuracy_fn(outputs, targets, "val")
        self.log(f'val_mean_accuracy', mean_accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        total_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_train_steps * 0.10)
        decay_steps = total_train_steps - warmup_steps
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps),
            'interval': 'step',
            'frequency': 1,
            'monitor': 'train_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

class CT_2DModel(CT_BaseModel):
    """ 2D model for CT scans. As input, the model expects a batch of images of size (B, 3, H, W).

    The model is composed of a backbone and 5 heads, one for each task.
    The backbone is a pretrained ResNet or EfficientNet model. The heads are composed of a linear layer,
    a SiLU activation function, a linear layer with 2, 2, 3, 3, 3 neurons respectively and
    a softmax activation function. The output of the model is a list of 6 tensors, one for each head. The last
    head, which corresponds to no injury, is computed from the injury probabilities of the other heads.
    The loss function is a weighted cross entropy loss.

    """
    def __init__(self, backbone=None, loss_weights=None, lr=1e-4):
        super().__init__(loss_weights=loss_weights, lr=lr)

        if backbone is None:
            backbone = config.BACKBONE
        try:
            self.backbone = config.backbone_dict[backbone]["model"](weights=config.backbone_dict[backbone]["weights"])
            if backbone in config.resnet_backbones:
                self.backbone.fc = nn.Identity()
            elif backbone in config.efficientnet_backbones:
                self.backbone.classifier = nn.Identity()
        except KeyError:
            raise KeyError(f"Backbone {backbone} not found. Please choose one of: {list(config.backbone_dict.keys())}")

        with torch.no_grad():
            output_size = self.backbone(torch.zeros(1, 3, *config.IMAGE_SIZE)).shape[1]

        self.heads = nn.ModuleList([
            # Bowel
            nn.Sequential(nn.Linear(output_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 2), nn.Softmax(dim=-1)),
            # Extra
            nn.Sequential(nn.Linear(output_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 2), nn.Softmax(dim=-1)),
            # Liver
            nn.Sequential(nn.Linear(output_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 3), nn.Softmax(dim=-1)),
            # Kidney
            nn.Sequential(nn.Linear(output_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 3), nn.Softmax(dim=-1)),
            # Spleen
            nn.Sequential(nn.Linear(output_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 3), nn.Softmax(dim=-1))
        ])

    def forward(self, x):
        # x.shape: (batch_size, 3, 256, 256)
        x = self.backbone(x)
        outputs = [head(x) for head in self.heads]
        # P[no_injury] = P[no_injury_bowel] * P[no_injury_extra] * P[no_injury_liver] * P[no_injury_kidney] * P[no_injury_spleen]
        no_injury = torch.prod(torch.stack([output[:, 0] for output in outputs], dim=-1),
                               dim=-1, keepdim=True)
        any_injury = 1-no_injury
        # output = [bowel, extra, liver, kidney, spleen, no_injury]
        # output.map(.shape) = [(batch_size, 2), (batch_size, 2), (batch_size, 3), (batch_size, 3), (batch_size, 3), (batch_size, 1)] 
        outputs = outputs + [any_injury]

        return outputs
    
# block args
blocks_args_str = [
    "r1_k3_s22_e1_i32_o16_se0.25", # This one is modified from s11 to s22, meaning stride 2
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]

efficientnet_dict = {
    "efficientnet-b0": {"width_mult": 1.0, "depth_mult": 1.0, "image_size": 224, "dropout_rate": 0.2, "dropconnect_rate": 0.2},
    "efficientnet-b1": {"width_mult": 1.0, "depth_mult": 1.1, "image_size": 240, "dropout_rate": 0.2, "dropconnect_rate": 0.2},
    "efficientnet-b2": {"width_mult": 1.1, "depth_mult": 1.2, "image_size": 260, "dropout_rate": 0.3, "dropconnect_rate": 0.2},
    "efficientnet-b3": {"width_mult": 1.2, "depth_mult": 1.4, "image_size": 300, "dropout_rate": 0.3, "dropconnect_rate": 0.2},
    "efficientnet-b4": {"width_mult": 1.4, "depth_mult": 1.8, "image_size": 380, "dropout_rate": 0.4, "dropconnect_rate": 0.2},
    "efficientnet-b5": {"width_mult": 1.6, "depth_mult": 2.2, "image_size": 456, "dropout_rate": 0.4, "dropconnect_rate": 0.2},
    "efficientnet-b6": {"width_mult": 1.8, "depth_mult": 2.6, "image_size": 528, "dropout_rate": 0.5, "dropconnect_rate": 0.2},
    "efficientnet-b7": {"width_mult": 2.0, "depth_mult": 3.1, "image_size": 600, "dropout_rate": 0.5, "dropconnect_rate": 0.2},
    "efficientnet-b8": {"width_mult": 2.2, "depth_mult": 3.6, "image_size": 672, "dropout_rate": 0.5, "dropconnect_rate": 0.2},
    "efficientnet-l2": {"width_mult": 4.3, "depth_mult": 5.3, "image_size": 800, "dropout_rate": 0.5, "dropconnect_rate": 0.2}
}

class CT_3DModel(CT_BaseModel):
    """ 3D model for CT scans. The idea of the model is that the backbone extracts a representation for each session,
    and then the aggregator combines the representations into a single representation. This representation is then
    used by the heads to predict the probabilities for each class.

    The model is composed of a backbone and 5 heads, one for each task.
    The backbone is a DenseNet model. The heads are composed of a linear layer,
    a SiLU activation function, a linear layer with 2, 2, 3, 3, 3 neurons respectively and
    a softmax activation function. The output of the model is a list of 6 tensors, one for each head. The last
    head, which corresponds to no injury, is computed from the injury probabilities of the other heads.
    The loss function is a weighted cross entropy loss.
    """
    def __init__(self, lr=1e-4, loss_weights=None):
        super().__init__(loss_weights=loss_weights, lr=lr)

        self.backbone  = monai.networks.nets.EfficientNet(blocks_args_str=blocks_args_str, spatial_dims=3, in_channels=1)
        self.backbone._fc = nn.Identity()

        rnn_input_size = 1280 + 1 # number of channels of backbone + aortic_hu
 
        hidden_size = config.RNN_HIDDEN_SIZE
        self.aggregator = nn.RNN(input_size=rnn_input_size, 
                                 hidden_size=hidden_size, 
                                 num_layers=2,
                                 nonlinearity = "relu",
                                 bidirectional=True, 
                                 batch_first=False)    
        
        head_input_size = hidden_size*2     

        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_input_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 2), nn.Softmax(dim=-1)),  # Bowel
            nn.Sequential(nn.Linear(head_input_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 2), nn.Softmax(dim=-1)),  # Extra
            nn.Sequential(nn.Linear(head_input_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 3), nn.Softmax(dim=-1)),  # Liver
            nn.Sequential(nn.Linear(head_input_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 3), nn.Softmax(dim=-1)),  # Kidney
            nn.Sequential(nn.Linear(head_input_size, config.HEAD_HIDDEN_SIZE), nn.SiLU(),
                          nn.Linear(config.HEAD_HIDDEN_SIZE, 3), nn.Softmax(dim=-1))  # Spleen
        ])

    def forward(self, x, aortic_hu):
        """ Forward pass of the model.

        Args:
            x: list of tensors of shape (b, C, H, W, D) with length B
            aortic_hu: list of tensors of shape (b, C) with length B
        
        Returns:
            outputs: list of tensors of shape (B, 2), (B, 2), (B, 3), (B, 3), (B, 3), (B, 1)

        Note: b = 1 (placeholder unitary batch size)
        """
        n_sessions = [len(volume) for volume in x]
#         x = max_shape_pad(x)
        x = torch.cat(x, dim=0) # concatenate along channel dim
        x = x.unsqueeze(1) # add channel dim
        x = self.backbone(x)
        x = torch.split(x, n_sessions, dim=0) # split along channel dim
        # concatenate the aortic_hu to the backbone output
        x = [torch.cat([volume, hu], dim=-1) for volume, hu in zip(x, aortic_hu)]
        # pass each volume to the aggregator separately and get the last hidden state
        x = torch.stack([self.aggregator(volume)[0][-1, :] for volume in x])

        outputs = [head(x) for head in self.heads]
        # P[no_injury] = Prod(P[no_injury_{task}] for task in tasks)
        no_injury = torch.prod(torch.stack([output[:, 0] for output in outputs], dim=-1),
                               dim=-1, keepdim=True)
        any_injury = 1-no_injury
        outputs += [any_injury]
        return outputs
    
    def _inference_fn(self, batch):
        images, aortic_hu, targets = batch
        outputs = self(images, aortic_hu)
        targets = split_targets(targets)
        return outputs, targets
    
    def predict_step(self, batch, batch_idx):
        x, aortic_hu = batch
        x = self.forward(x, aortic_hu)
        x = torch.concat(x, 1)
        return x

    def configure_optimizers(self):
        # NOTE: We use SGD as Adam has issues with float16
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                  total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]