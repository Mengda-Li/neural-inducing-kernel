import torch
import torchvision.models as models
from torch import optim, nn
import lightning as L
from kernel_functions import KernelFunction, ExponentialKernel
from loss import Loss, SSIMLoss


class Resnet_kernel_w(L.LightningModule):
    def __init__(
        self,
        num_induce_pt: int = 10,
        frozen_resnet: bool = True,
        kernel: KernelFunction = ExponentialKernel,
        loss: Loss = SSIMLoss,
    ):
        super().__init__()
        self.num_induce_pt = num_induce_pt
        self.frozen_resnet = frozen_resnet
        self.kernel = kernel
        self.loss = loss

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        if self.frozen_resnet:
            self.feature_extractor.eval()

        # use the pretrained model to extract inducing points (each point is (R,G,B))
        # num_target_dim = 3 * self.num_induce_pt

        # the Alpha weights for each kernel function
        num_target_dim = (
            6 * self.num_induce_pt
        )  # 3 * self.num_induce_pt + 3 * self.num_induce_pt
        self.induce_pt_extractor = nn.Linear(num_filters, num_target_dim)

    def forward(self, x):
        if self.frozen_resnet:
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        x = self.induce_pt_extractor(representations)

        induce_pt = x[:, : -3 * self.num_induce_pt]
        kernel_weight = x[:, -3 * self.num_induce_pt :]

        batch_size = x.shape[0]
        induce_pt = induce_pt.reshape((batch_size, 3, self.num_induce_pt))
        kernel_weight = kernel_weight.reshape((batch_size, 3, self.num_induce_pt))

        return induce_pt, kernel_weight

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        original, target, normalized = batch

        induce_pt, kernel_weight = self.forward(normalized)
        kernel = self.kernel()
        f = kernel(induce_pt, kernel_weight)
        enhanced = f(original)

        loss_f = self.loss()
        loss_value = loss_f(enhanced, target)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss_value)
        return loss_value

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        original, target, normalized = batch

        induce_pt, kernel_weight = self.forward(normalized)
        kernel = self.kernel()
        f = kernel(induce_pt, kernel_weight)
        enhanced = f(original)

        loss_f = self.loss()
        loss_value = loss_f(enhanced, target)
        self.log("val_loss", loss_value)

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=1e-3)
        return optimizer
