import glob
from PIL import Image
import numpy as np
import torch
import cv2
from skimage.transform import resize

import models


if __name__ == "__main__":
    device = torch.device("cuda")
    img_embedding_size = 128
    pos_embedding_size = 128
    decoder_hidden_size = 128
    batch_size = 1

    image_paths = glob.glob("images/*.jfif")
    images = [np.asarray(Image.open(image_path)) for image_path in image_paths]
    for i in range(len(images)):
        image = images[i]
        dimensions = image.shape[:2]
        ratio = min([min(d, 480) / d for d in dimensions])
        scaled_dimensions = [int(ratio * d) for d in dimensions]
        image = resize(image, (*scaled_dimensions, 3), anti_aliasing=True)
        image = image[:, :, [2, 1, 0]]
        image = np.transpose(image, (2, 0, 1))
        images[i] = image

    encoder = models.Encoder(
        embedding_size=img_embedding_size
    ).to(device)
    decoder = models.Decoder(
        img_embedding_size=img_embedding_size,
        pos_embedding_size=pos_embedding_size,
        hidden_size=decoder_hidden_size
    ).to(device)

    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=0.001)
    encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(encoder_opt, gamma=0.99999)
    decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(decoder_opt, gamma=0.99999)

    for i in range(100000000):
        image_idxs = np.random.randint(low=0, high=len(images), size=batch_size)
        image = [images[idx] for idx in image_idxs]
        shape_idx = np.random.randint(low=0, high=batch_size)
        image_shape = image[shape_idx].shape
        if batch_size > 1:
            image = [resize(img, image_shape, anti_aliasing=True) for img in image]
        image = np.stack(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32, device=device)

        mean, log_std = encoder(image)
        std = log_std.exp()

        sample_dist = torch.distributions.Normal(loc=mean, scale=std)
        sample = sample_dist.sample()

        height, width = image_shape[1:3]
        # TODO sample instead of mean
        decoded_image = decoder(mean, height, width)

        kl_loss = 0.5 * (mean ** 2 + std ** 2 - 2 * log_std - 1).sum(dim=1)
        kl_loss = kl_loss.mean()
        l2_loss = ((image - decoded_image) ** 2).sum(dim=1).mean(dim=(1, 2))
        l2_loss = l2_loss.mean()
        # TODO use kl_loss
        loss = 0 * kl_loss + l2_loss
        print(kl_loss.item(), l2_loss.item(), encoder_opt.state_dict()["param_groups"][0]["lr"])

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()
        encoder_scheduler.step()
        decoder_scheduler.step()

        if i % 40 == 0:
            np_image = decoded_image[shape_idx].permute((1, 2, 0)).detach().cpu().numpy()
            # np_image = image[0].permute((1, 2, 0)).detach().cpu().numpy()
            cv2.imshow('image', np_image)
            cv2.waitKey(1)
