#optimzed for kaggle GPU run


def main():

    import sys
    from pathlib import Path

    # ensure project imports work when run as a script
    PROJECT_ROOT = Path.cwd()

    if PROJECT_ROOT.name != "economic-satellite-project":
        PROJECT_ROOT = PROJECT_ROOT / "economic-satellite-project"
        sys.path.insert(0, str(PROJECT_ROOT))

    from models.resnet_frozen import ResNetRegressor
    import pandas as pd
    from models.dataset import CensusDataset
    from torch.utils.data import DataLoader
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from sklearn.metrics import mean_absolute_error
    from models.collate import collate_fn
    from src.splitting import split_by_tract

    # instantiate frozen resnet and get its preprocessing transforms
    model = ResNetRegressor()
    transform = model.weights.transforms()

    # device autodetection: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model.to(device)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # load CSV of tiles and GEOIDs
    df = pd.read_csv(
        "/kaggle/input/datasets/braydenwinnicki/processed-csv-tracts-tiles/processed_ct_tracts_tiles.csv",
        dtype={"GEOID": str}
    )

    df.columns = df.columns.str.strip()

    # split train/test by tract so all tiles from the same tract stay together
    df_train, df_test = split_by_tract(df, test_size=0.20, random_state=42)

    # compute normalization stats from the training tracts only
    train_tract_labels = df_train.drop_duplicates(subset="GEOID")["median_income"]
    mean_income = train_tract_labels.mean()
    std_income = train_tract_labels.std()
    if std_income == 0:
        std_income = 1.0

    df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

    # create dataset using cached images
    train_dataset = CensusDataset(
        df_train,
        "/kaggle/input/datasets/braydenwinnicki/census-images-cache-pt/census_images_cache.pt",
        dtype={"GEOID": str}
    )

    # choose batch size based on device to avoid MPS OOM
    default_batch = 64
    if device.type == "mps":
        default_batch = 8

    # DataLoader with multiple workers for speed and persistent workers to reduce overhead
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=default_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=True,
        pin_memory=pin_memory,
    )

    criterion = nn.MSELoss()  # mean squared loss
    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
    
)

    scaler = torch.cuda.amp.GradScaler()

    # training loop
    epochs = 30

    model.train()  # enable training behavior

    for epoch in range(epochs):

        total_loss = 0

        for images, mask, incomes, geoids in train_loader:

            # move tensors to device; cached images are float32
            images = images.float().to(device)
            mask = mask.to(device)
            incomes = incomes.to(device)

            # forward pass
            with torch.cuda.amp.autocast():
                predictions = model(images, mask)
                loss = criterion(predictions.squeeze(), incomes.float())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"train Epoch {epoch+1}: {avg_loss:.4f}")

    # save the trained model weights
    torch.save(model.state_dict(), PROJECT_ROOT / "models" / "resnet18_frozen.pth")

    print("Saved model to models/resnet18_frozen.pth")


if __name__ == "__main__":
    main()
