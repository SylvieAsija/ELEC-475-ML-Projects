import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class encoder_classify:
    encoder = nn.Sequential(*list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
    simple_classification = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )


class CustomNetwork(nn.Module):
    def __init__(self, encoder, classification):
        super(CustomNetwork, self).__init__()

        # Load pre-trained ResNet-18 model
        self.encoder = encoder
        if self.encoder is None:
            self.encoder = encoder_classify.encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.mse_loss = nn.BCEWithLogitsLoss()

        self.classification = classification
        if self.classification is None:
            self.classification = encoder_classify.simple_classification
            self.init_classification_weights(mean=0.0, std=0.1)

    def init_classification_weights(self, mean, std):
        for param in self.classification.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def encode(self, X):
        return self.encoder(X)

    def classify(self, X):
        return self.classification(X)

    def forward(self, x):
        if self.training:
            x = self.encoder(x)
            output = self.classify(x)
            return output
        else:
            x = self.encoder(x)
            output = self.classify(x)
            class_probabilities = torch.sigmoid(output)
            return class_probabilities
