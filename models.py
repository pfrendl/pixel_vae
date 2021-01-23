from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, embedding_size: int):
        super(Encoder, self).__init__()
        self.cl0 = nn.Conv2d(3, 8, 3, 2, 2)
        self.cl1 = nn.Conv2d(8, 16, 3, 2, 2)
        self.cl2 = nn.Conv2d(16, 32, 3, 2, 2)
        self.cl3 = nn.Conv2d(32, 32, 3, 2, 2)
        self.cl4 = nn.Conv2d(32, 32, 3, 2, 2)
        self.cl5 = nn.Conv2d(32, 64, 3, 2, 2)
        self.cl6 = nn.Conv2d(64, embedding_size, 3, 2, 2)
        self.fc0 = nn.Linear(embedding_size, embedding_size)
        self.mean = nn.Linear(embedding_size, embedding_size)
        self.std = nn.Linear(embedding_size, embedding_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.relu(self.cl0(x))
        x = torch.relu(self.cl1(x))
        x = torch.relu(self.cl2(x))
        x = torch.relu(self.cl3(x))
        x = torch.relu(self.cl4(x))
        x = torch.relu(self.cl5(x))
        x = torch.relu(self.cl6(x))
        # print("x.shape ---", x.shape)
        x = torch.mean(x, dim=(2, 3))
        x = torch.relu(self.fc0(x))
        mean = self.mean(x)
        log_std = self.std(x)
        return mean, log_std


class SkipBlock(nn.Module):
    def __init__(self, size: int, scaling: float):
        super(SkipBlock, self).__init__()
        scaled_size = int(scaling * size)
        self.fc0 = nn.Linear(size, scaled_size)
        self.fc1 = nn.Linear(scaled_size, scaled_size)
        self.fc2 = nn.Linear(scaled_size, size)

    def forward(self, x):
        y = torch.relu(self.fc0(x))
        y = torch.relu(self.fc1(y))
        y = self.fc2(y)
        return x + y


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super(PositionEmbedding, self).__init__()
        self.fc0 = nn.Linear(2, 128)
        self.fc1 = nn.Linear(128, embedding_size)

    def forward(self, height: int, width: int, device: torch.device) -> Tensor:
        dim0_range = torch.linspace(start=-2, end=2, steps=height).to(device)
        dim1_range = torch.linspace(start=-2, end=2, steps=width).to(device)
        grids = torch.meshgrid(dim0_range, dim1_range)
        pixel_coords = torch.stack(grids, dim=-1)  # (height, width, 2)
        x = pixel_coords.view((-1, 2))  # (height * width, 2)
        x = torch.relu(self.fc0(x))  # (height * width, embedding_size)
        x = self.fc1(x)  # (height * width, embedding_size)
        return x


class Decoder(nn.Module):
    def __init__(self, img_embedding_size: int, pos_embedding_size: int, hidden_size: int):
        super(Decoder, self).__init__()
        self.position_embedding = PositionEmbedding(pos_embedding_size)
        self.fc0 = nn.Linear(img_embedding_size + pos_embedding_size, hidden_size)
        self.sb0 = SkipBlock(hidden_size, 1)
        self.sb1 = SkipBlock(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 3)

    def forward(self, embedding: Tensor, height: int, width: int) -> Tensor:
        x = embedding  # (batch_size, embedding_size)

        position_embeddings = self.position_embedding(height, width, embedding.device)  # (height * width, 2)
        position_embeddings = position_embeddings[None, :, :].expand(x.shape[0], -1, -1)  # (batch_size, height * width, 2)
        x = x[:, None, :].expand(-1, position_embeddings.shape[1], -1)  # (batch_size, height * width, embedding_size)

        x = torch.cat([x, position_embeddings], axis=2)  # (batch_size, height * width, embedding_size + 2)
        x = x.view((-1, x.shape[2]))  # (batch_size * height * width, embedding_size + 2)

        x = torch.relu(self.fc0(x))  # (batch_size * height * width, embedding_size)
        x = self.sb0(x)
        x = self.sb1(x)
        x = self.fc1(x)  # (batch_size * height * width, 3)

        x = x.view((embedding.shape[0], height, width, 3))  # (batch_size, height, width, 3)
        x = x.permute((0, 3, 1, 2))  # (batch_size, 3, height, width)

        return x
