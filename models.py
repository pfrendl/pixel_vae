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
        mean = self.mean(x)
        log_std = self.std(x)
        return mean, log_std


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super(PositionEmbedding, self).__init__()
        self.fc0 = nn.Linear(2, 128)
        self.fc1 = nn.Linear(128, embedding_size)
        self.log_frequency = nn.Parameter(10 * torch.randn((1, embedding_size)))
        self.amplitude = nn.Parameter(10 * torch.ones((1, embedding_size)))
        self.phase = nn.Parameter(torch.randn((1, embedding_size)))

    def forward(self, height: int, width: int, device: torch.device) -> Tensor:
        dim0_range = torch.linspace(start=-1, end=1, steps=height).to(device)
        dim1_range = torch.linspace(start=-1, end=1, steps=width).to(device)
        grids = torch.meshgrid(dim0_range, dim1_range)
        pixel_coords = torch.stack(grids, dim=-1)  # (height, width, 2)
        x = pixel_coords.view((-1, 2))  # (height * width, 2)
        x = torch.relu(self.fc0(x))  # (height * width, embedding_size)
        x = self.fc1(x)  # (height * width, embedding_size)
        frequency = self.log_frequency.exp()
        amplitude = self.amplitude
        phase = self.phase
        x = amplitude * (frequency * x).sin() + phase
        return x


class Decoder(nn.Module):
    def __init__(self, img_embedding_size: int, pos_embedding_size: int, hidden_size: int):
        super(Decoder, self).__init__()
        self.position_embedding = PositionEmbedding(pos_embedding_size)
        self.fc0 = nn.Linear(img_embedding_size + pos_embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 3)

    def forward(self, embedding: Tensor, height: int, width: int) -> Tensor:
        x = embedding  # (batch_size, embedding_size)

        batch_size = embedding.shape[0]
        position_embeddings = self.position_embedding(height, width, embedding.device)  # (height * width, 2)
        position_embeddings = position_embeddings[None, :, :].expand(batch_size, -1, -1)  # (batch_size, height * width, 2)
        x = x[:, None, :].expand(-1, position_embeddings.shape[1], -1)  # (batch_size, height * width, embedding_size)

        x = torch.cat([x, position_embeddings], axis=2)  # (batch_size, height * width, embedding_size + 2)
        x = x.view((-1, x.shape[2]))  # (batch_size * height * width, embedding_size + 2)

        x = torch.relu(self.fc0(x))  # (batch_size * height * width, embedding_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # (batch_size * height * width, 3)

        x = x.view((embedding.shape[0], height, width, 3))  # (batch_size, height, width, 3)
        x = x.permute((0, 3, 1, 2))  # (batch_size, 3, height, width)

        return x
