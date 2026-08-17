import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Standard ResNet Basic Block with Identity Shortcut Connection"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, is_plain: bool = False):
        super(BasicBlock, self).__init__()
        self.is_plain = is_plain

        # Main convolutional branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut branch (Projection shortcut used when dimension or stride changes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual Addition: F(x) + x (Skipped if evaluating a Plain Network)
        if not self.is_plain:
            out += self.shortcut(identity)

        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, 
        block: type, 
        layers: list, 
        num_classes: int = 1000, 
        in_channels: int = 3, 
        is_plain: bool = False
    ):
        """
        Deep Residual Learning for Image Recognition (He et al., 2015)
        
        Args:
            block: Block class to instantiate (e.g., BasicBlock).
            layers: List defining the number of blocks in each stage (e.g., [2, 2, 2, 2] for ResNet-18).
            num_classes: Number of classification output targets.
            in_channels: Input image channel depth (3 for RGB).
            is_plain: If True, disables residual shortcuts to construct a Plain Net for comparison.
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.is_plain = is_plain

        # 1. Stem (Initial Feature Extraction)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. Residual Stages (Contracting & Deep Feature Representation)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 3. Global Pooling & Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        # First block in stage handles downsampling via stride
        layers.append(block(self.in_channels, out_channels, stride=stride, is_plain=self.is_plain))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, is_plain=self.is_plain))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits


# Helper functions to build ResNet-18 variants
def build_resnet18(num_classes: int = 1000, is_plain: bool = False) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, is_plain=is_plain)


# ==========================================
# End-to-End Pipeline & Degradation Test
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_CLASSES = 10  # e.g., CIFAR-10 or custom target dataset
    HEIGHT, WIDTH = 224, 224

    # 1. Instantiate standard ResNet-18 (With Identity Shortcut Connections)
    resnet18 = build_resnet18(num_classes=NUM_CLASSES, is_plain=False)

    # 2. Instantiate Plain-18 Network (Without Shortcut Connections) for Degradation Analysis
    plain18 = build_resnet18(num_classes=NUM_CLASSES, is_plain=True)

    # Synthetic Batch of Input Images (Batch Size x Channels x Height x Width)
    dummy_input = torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH)
    dummy_labels = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH_SIZE,))

    # Setup Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Forward Pass Comparison
    resnet_logits = resnet18(dummy_input)
    plain_logits = plain18(dummy_input)

    print("ResNet-18 Output Logits Shape:", resnet_logits.shape)  # Expected: [16, 10]
    print("Plain-18 Output Logits Shape: ", plain_logits.shape)   # Expected: [16, 10]

    # Training Step Execution
    loss = criterion(resnet_logits, dummy_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"ResNet-18 Training Step Successful. Cross-Entropy Loss: {loss.item():.4f}")
