import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.init(entity="alinajibpour",project="mlops")

class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)  # Adjusted to match flattened output size

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        print(x.shape, "test shape")
        x = self.dropout(x)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        # Log non-scalar tensor
        self.logger.experiment.log({'logits': wandb.Histogram(preds.detach().cpu().numpy())})

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        img, target = batch
        preds = self(img)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def prepare_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root=r'C:\Users\ra59xaf\Desktop\my_project', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=r'C:\Users\ra59xaf\Desktop\my_project', train=False, download=True, transform=transform)

    # Limit training data to 20%
    train_size = int(0.2 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    model = MyAwesomeModel()
    train_loader, val_loader, test_loader = prepare_data()

    # Wandb logger
    wandb_logger = WandbLogger(entity="ali-najibpour",project="mlops")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=10,  # Reduce the number of epochs from 1000 to 10
        max_steps=1000,  # Optional: limit the number of training steps
        default_root_dir='C:/Users/ra59xaf/Desktop/my_project',  # Specify a directory to save logs and checkpoints
        limit_train_batches=1.0,  # Use only 100% of the training data (since we already reduced it to 20%)
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=wandb_logger
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
