import torch.nn as nn

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3))
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3))
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        out = self.pad2(out)
        out += x
        return out


class encoder_classification:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )

    modded_classification_head = nn.Sequential(
        res_block(in_channels=512, out_channels=512),
        res_block(in_channels=512, out_channels=512),
        nn.Flatten(),
        nn.Linear(8192, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 100)
    )

    classification_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 100)
    )


class network(nn.Module):
    def __init__(self, encoder, classification):
        super(network, self).__init__()
        self.encoder = encoder
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        # need access to these intermediate encoder steps
        # for the encoder computation
        encoder_list = list(encoder.children())
        self.encoder_stage_1 = nn.Sequential(*encoder_list[:4])  # input -> relu1_1
        self.encoder_stage_2 = nn.Sequential(*encoder_list[4:11])  # relu1_1 -> relu2_1
        self.encoder_stage_3 = nn.Sequential(*encoder_list[11:18])  # relu2_1 -> relu3_1
        self.encoder_stage_4 = nn.Sequential(*encoder_list[18:31])  # relu3_1 -> relu4_1

        self.mse_loss = nn.MSELoss()

        self.classification = classification
        if self.classification is None:
            self.classification = encoder_classification.modded_classification_head
            # self.classification = encoder_classification.classification_head
            self.init_classification_weights(mean=0.0, std=0.1)

    def init_classification_weights(self, mean, std):
        print("initing weights!")
        for param in self.classification.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def encode(self, X):
        return self.encoder(X)

    def classify(self, X):
        return self.classification(X)

    def forward(self, X):
        if self.training:
            # encode the input
            X = self.encode(X)

            # pass through classification head
            output = self.classify(X)

            return output
        else:
            # encode the input
            X = self.encode(X)

            # pass through classification head
            output = self.classify(X)

            # apply softmax activation for class probabilities
            class_probabilities = nn.functional.softmax(output, dim=1)

            return class_probabilities
