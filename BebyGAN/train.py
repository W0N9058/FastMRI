import argparse
import os  # os 모듈을 임포트 추가
import torch
from torch import nn, optim
from utils.data.load_data import create_data_loaders  # get_dataloader 대신 create_data_loaders 사용
from network import Network
from config import config

def get_dataloader(data_path, args):
    train_loader = create_data_loaders(
        data_path=os.path.join(data_path, 'train'),
        args=args,
        shuffle=True,
        isforward=False
    )
    val_loader = create_data_loaders(
        data_path=os.path.join(data_path, 'val'),
        args=args,
        shuffle=False,
        isforward=True
    )
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=config.DATALOADER.IMG_PER_GPU, help='Batch size')
    parser.add_argument('--max_key', type=str, default='max', help='Max key for normalization')
    parser.add_argument('--target_key', type=str, default='target', help='Target key in the dataset')
    parser.add_argument('--input_key', type=str, default='kspace', help='Input key in the dataset')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Sample rate for the dataset')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    args = parser.parse_args()

    train_loader, val_loader = get_dataloader(args.dataset_path, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = Network(config)
    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=config.SOLVER.G_BASE_LR, betas=(config.SOLVER.G_BETA1, config.SOLVER.G_BETA2))

    criterion = nn.MSELoss()

    for epoch in range(args.num_epochs):  # args.num_epochs로 변경
        network.train()
        for i, data in enumerate(train_loader):
            # 디버깅을 위한 데이터 출력
            print(f"Data at index {i}: {data}")

            try:
                # 데이터의 구조에 맞게 수정
                if len(data) == 5:
                    kspace, target, max_key, fname, slice_idx = data
                    mask = None  # mask가 없는 경우
                else:
                    kspace, target, mask, max_key, fname, slice_idx = data

                # 디버깅을 위한 개별 데이터 출력
                print(f"kspace shape: {kspace.shape}")
                print(f"target shape: {target.shape}")
                print(f"mask shape: {mask.shape}" if mask is not None else "mask: None")
                print(f"max_key: {max_key}")
                print(f"fname: {fname}")
                print(f"slice_idx: {slice_idx}")

                # 데이터 전처리
                kspace, target = kspace.to(device), target.to(device)
                if mask is not None:
                    mask = mask.to(device)

                optimizer.zero_grad()
                # mask가 없는 경우 네트워크 호출 방법 변경
                outputs, _ = network(kspace, mask) if mask is not None else network(kspace)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                if i % config.LOG_PERIOD == 0:
                    print(f"Epoch [{epoch}/{args.num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

            except ValueError as e:
                print(f"ValueError: {e}")
                # 데이터의 형식 확인을 위해 출력
                print(f"data contents: {data}")
                # 디버깅 후 중단
                break

        if epoch % config.VAL.PERIOD == 0:
            network.eval()
            with torch.no_grad():
                val_loss = 0
                for i, data in enumerate(val_loader):
                    # 데이터의 구조에 맞게 수정
                    if len(data) == 5:
                        kspace, target, max_key, fname, slice_idx = data
                        mask = None  # mask가 없는 경우
                    else:
                        kspace, target, mask, max_key, fname, slice_idx = data

                    kspace, target = kspace.to(device), target.to(device)
                    if mask is not None:
                        mask = mask.to(device)

                    outputs, _ = network(kspace, mask) if mask is not None else network(kspace)  # mask가 없으면 제외
                    loss = criterion(outputs, target)
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                print(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")

        if epoch % config.SAVE_PERIOD == 0:
            torch.save(network.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()