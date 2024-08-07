import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple
import random
import fastmri
from fastmri.data import transforms as T
from utils.data.transforms import DataTransform, get_augmentor

# U-Net 클래스 정의 (생략된 부분은 이전과 동일)
class Unet(nn.Module):
    def __init__(self, in_chans, out_chans, chans=32, num_pool_layers=4, drop_prob=0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = nn.functional.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = nn.functional.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)

class TransposeConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)

# VarNet 클래스 정의 (생략된 부분은 이전과 동일)
class NormUnet(nn.Module):
    def __init__(self, chans, num_pools, in_chans=2, out_chans=2, drop_prob=0.0):
        super().__init__()
        self.unet = Unet(in_chans=in_chans, out_chans=out_chans, chans=chans, num_pool_layers=num_pools, drop_prob=drop_prob)

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = torch.nn.functional.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x

class SensitivityModel(nn.Module):
    def __init__(self, chans, num_pools, in_chans=2, out_chans=2, drop_prob=0.0):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools, in_chans=in_chans, out_chans=out_chans, drop_prob=drop_prob)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeezed_mask = mask[:, 0, 0, :, 0].bool()
        cent = squeezed_mask.shape[1] // 2
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(2 * torch.min(left, right), torch.ones_like(left))
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        x = T.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x

class VarNet(nn.Module):
    def __init__(self, num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4):
        super().__init__()
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList([VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)])

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        height = result.shape[-2]
        width = result.shape[-1]
        return result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]

class VarNetBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(self, current_kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps)
        return current_kspace - soft_dc - model_term

# 판별자 정의
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# SliceData 클래스 정의
class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            print(f"Found {len(image_files)} image files.")
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname, input_key)
                self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

        kspace_files = list(Path(root / "kspace").iterdir())
        print(f"Found {len(kspace_files)} k-space files.")
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname, 'kspace')  # k-space 파일의 키는 'kspace'
            if num_slices > 0:
                self.kspace_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            else:
                print(f"Warning: No slices found in k-space file {fname}")

        print(f"Loaded {len(self.image_examples)} image examples and {len(self.kspace_examples)} k-space examples.")

    def _get_metadata(self, fname, key):
        num_slices = 0  # num_slices 변수를 초기화
        with h5py.File(fname, "r") as hf:
            if key in hf.keys():
                num_slices = hf[key].shape[0]
            else:
                print(f"Warning: {fname} does not contain {key}")
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf['kspace'][dataslice]  # k-space 파일의 키는 'kspace'
            mask = np.array(hf["mask"])

        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)

def custom_collate_fn(batch):
    mask, input, target, attrs, fname, dataslice = zip(*batch)

    # Determine the maximum slice, height, width, and the last dimension in this batch
    max_slices = max(x.shape[0] for x in input)
    max_height = max(x.shape[-3] for x in input)
    max_width = max(x.shape[-2] for x in input)
    max_last_dim = max(x.shape[-1] for x in input)

    # Pad images and masks to the maximum size
    padded_input = [torch.nn.functional.pad(x, (0, max_last_dim - x.shape[-1], 0, max_width - x.shape[-2], 0, max_height - x.shape[-3], 0, max_slices - x.shape[0])) for x in input]
    padded_mask = [torch.nn.functional.pad(x, (0, max_last_dim - x.shape[-1], 0, max_width - x.shape[-2], 0, max_height - x.shape[-3], 0, max_slices - x.shape[0])) for x in mask]

    return (torch.stack(padded_mask, dim=0),
            torch.stack(padded_input, dim=0),
            torch.stack(target, dim=0),
            attrs,
            fname,
            dataslice)

def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    current_epoch_fn = lambda: 0  # Placeholder, you should use the actual epoch function if available
    augmentor = get_augmentor(args, current_epoch_fn)
    
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
        
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_, augmentor=augmentor),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )
    return data_loader

# 학습 설정
class Args:
    def __init__(self):
        self.GPU_NUM = 0
        self.batch_size = 2  # 배치 크기를 줄임
        self.num_epochs = 100
        self.lr = 0.0002
        self.report_interval = 10
        self.data_path_train = Path('/home/Data/train/')
        self.data_path_val = Path('/home/Data/val/')
        self.input_key = 'image_input'
        self.target_key = 'image_label'
        self.max_key = 'max'
        self.seed = 42
        self.cascade = 1
        self.chans = 9
        self.sens_chans = 4
        self.aug_on = True
        self.aug_schedule = 'exp'
        self.aug_delay = 0
        self.aug_strength = 0.0
        self.aug_exp_decay = 5.0
        self.aug_interpolation_order = 1
        self.aug_upsample = False
        self.aug_upsample_factor = 2
        self.aug_upsample_order = 1
        self.aug_weight_translation = 1.0
        self.aug_weight_rotation = 1.0
        self.aug_weight_shearing = 1.0
        self.aug_weight_scaling = 1.0
        self.aug_weight_rot90 = 1.0
        self.aug_weight_fliph = 1.0
        self.aug_weight_flipv = 1.0
        self.aug_max_translation_x = 0.125
        self.aug_max_translation_y = 0.125
        self.aug_max_rotation = 180.0
        self.aug_max_shearing_x = 15.0
        self.aug_max_shearing_y = 15.0
        self.aug_max_scaling = 0.25
        self.max_train_resolution = None  # 여기에 max_train_resolution 속성 추가

args = Args()

# 데이터 로더 준비
train_loader = create_data_loaders(args.data_path_train, args, shuffle=True)
val_loader = create_data_loaders(args.data_path_val, args)

# 모델 초기화
G = VarNet(num_cascades=2, sens_chans=8, sens_pools=4, chans=18, pools=4)  # VarNet을 생성자로 사용
input_size = 384 * 384  # 데이터 크기에 맞게 설정
hidden_size = 256
output_size = 1
D = Discriminator(input_size, hidden_size, output_size)

# 손실 함수 및 옵티마이저 설정
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

# 학습 루프
for epoch in range(args.num_epochs):
    for i, data in enumerate(train_loader):
        mask, kspace, target, _, _, _ = data

        # 진짜 데이터 준비
        real_images = target.view(args.batch_size, -1)  # 판별자에 맞추기 위해 2D 텐서로 변환
        real_labels = torch.ones(args.batch_size, 1)
        fake_labels = torch.zeros(args.batch_size, 1)

        # 판별자 학습
        outputs = D(real_images)
        D_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        fake_images = G(kspace, mask)  # VarNet에 맞는 입력 형식 사용
        outputs = D(fake_images.reshape(args.batch_size, -1))  # 판별자에 맞추기 위해 2D 텐서로 변환
        D_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        D_loss = D_loss_real + D_loss_fake
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # 생성자 학습
        fake_images = G(kspace, mask)  # VarNet에 맞는 입력 형식 사용
        outputs = D(fake_images.reshape(args.batch_size, -1))  # 판별자에 맞추기 위해 2D 텐서로 변환
        G_loss = criterion(outputs, real_labels)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if (i + 1) % args.report_interval == 0:
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}')

print("Training finished!")
