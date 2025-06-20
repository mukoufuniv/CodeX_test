import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from .models import SimpleConvNet, WiderConvNet
from .attention import AttentionHook, gram_matrix

class Distiller:
    def __init__(self, model_cls=SimpleConvNet, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_cls().to(self.device)
        self.hook = AttentionHook()
        # Attach hook to last conv layer
        last_conv = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)][-1]
        last_conv.register_forward_hook(self.hook)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def _real_loader(self, batch_size=64):
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        return DataLoader(train, batch_size=batch_size, shuffle=True)

    def init_synthetic(self, n_images=10):
        # Start with random synthetic data
        self.synthetic_data = torch.randn(n_images, 1, 28, 28, requires_grad=True, device=self.device)
        self.synthetic_targets = torch.randint(0, 10, (n_images,), device=self.device)
        self.syn_optimizer = optim.SGD([self.synthetic_data], lr=0.1)

    def distill_step(self, real_batch):
        self.optim.zero_grad()
        self.hook.clear()

        inputs, targets = real_batch[0].to(self.device), real_batch[1].to(self.device)
        outputs = self.model(inputs)
        loss_real = self.criterion(outputs, targets)
        attn_real = self.hook.maps[-1]
        self.hook.clear()
        loss_real.backward()

        # Synthetic
        self.syn_optimizer.zero_grad()
        syn_outputs = self.model(self.synthetic_data)
        loss_syn = self.criterion(syn_outputs, self.synthetic_targets)
        attn_syn = self.hook.maps[-1]
        gm_real = gram_matrix(attn_real)
        gm_syn = gram_matrix(attn_syn)
        attn_loss = nn.functional.mse_loss(gm_syn, gm_real)
        total_loss = loss_syn + attn_loss
        total_loss.backward()
        self.syn_optimizer.step()

    def distill(self, steps=10):
        loader = self._real_loader()
        iterator = iter(loader)
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            self.distill_step(batch)

    def synthetic_dataset(self):
        data = self.synthetic_data.detach().cpu()
        targets = self.synthetic_targets.cpu()
        return TensorDataset(data, targets)


def evaluate_cross_architecture(distilled_ds, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = WiderConvNet().to(device)
    loader = DataLoader(distilled_ds, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optim_ = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(2):  # quick training
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optim_.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim_.step()
    return model

if __name__ == '__main__':
    distiller = Distiller()
    distiller.init_synthetic()
    distiller.distill()
    distilled_ds = distiller.synthetic_dataset()
    evaluate_cross_architecture(distilled_ds)
