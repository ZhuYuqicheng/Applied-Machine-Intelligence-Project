import argparse
from train_with_sampling import *
from DataLoader import *
from inference import *


def main(
        epoch: int = 500,
        k: int = 60,
        batch_size: int = 100,
        training_length=72,
        forecast_window=24,
        train_csv="Data/data_train_1.csv",
        test_csv="Data/data_pe_test_uni_1.csv",
        path_to_save_model="save_model/",
        path_to_save_loss="save_loss/",
        path_to_save_predictions="save_predictions/",
        device="cpu"
):
    # clean_directory()

    train_dataset = SpotPriceDataset(data_list_path=train_csv, training_length=training_length,
                                     forecast_window=forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = SpotPriceDataset(data_list_path=test_csv, training_length=training_length,
                                    forecast_window=forecast_window, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    best_model = transformer(train_dataloader, epoch, k, path_to_save_model, path_to_save_loss,
                             path_to_save_predictions, device, begin_at=0)
    # best_model = f"best_train_182.pth"
    inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--path_to_save_model", type=str, default="save_model/")
    parser.add_argument("--path_to_save_loss", type=str, default="save_loss/")
    parser.add_argument("--path_to_save_predictions", type=str, default="save_predictions/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k=args.k,
        batch_size=args.batch_size,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,
        device=args.device,
    )
