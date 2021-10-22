import logging
import math
import random

from joblib import load
from tqdm import tqdm
from helpers import *
from model import Transformer
from plot import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False


def transformer(dataloader, EPOCH, k, path_to_save_model, path_to_save_loss, path_to_save_predictions,
                device, begin_at=0):
    device = torch.device(device)

    model = Transformer(dropout=0.1).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    if begin_at != 0:
        model.load_state_dict(torch.load(path_to_save_model + 'best_train_{}.pth'.format(begin_at - 1)))
        optimizer.load_state_dict(torch.load(path_to_save_model + 'optimizer_{}.pth'.format(begin_at - 1)))
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(begin_at, EPOCH + 1):
        print('EPOCH:', epoch)
        train_loss = 0
        val_loss = 0

        # TRAIN -- TEACHER FORCING
        model.train()
        for index_in, index_tar, _input, target in tqdm(dataloader, 'BATCH'):

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            optimizer.zero_grad()
            src = _input.permute(1, 0, 2).double().to(device)[:-1, :, :]  # torch.Size([L-1, B, 7])
            target = _input.permute(1, 0, 2).double().to(device)[1:, :, :]  # src shifted by 1.
            sampled_src = src[:1, :, :]  # t0 torch.Size([1, B, F])

            for i in range(len(target) - 1):

                prediction = model(sampled_src, device)  # torch.Size([1~W, B, 1])
                # for p1, p2 in zip(params, model.parameters()):
                #     if p1.data.ne(p2.data).sum() > 0:
                #         ic(False)
                # ic(True)
                # ic(i, sampled_src[:,:,0], prediction)
                # time.sleep(1)
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                if i < 24:  # One day, enough data to make inferences about cycles
                    prob_true_val = True
                else:
                    # coin flip
                    v = k / (k + math.exp(
                        epoch / k))  # probability of heads/tails depends on the epoch, evolves with time.
                    prob_true_val = flip_from_probability(
                        v)  # starts with over 95 % probability of true val for each flip in epoch 0.
                    # if using true value as new value
                if prob_true_val:  # Using true value as next value
                    sampled_src = torch.cat((sampled_src.detach(), src[i + 1, :, :].unsqueeze(0).detach()))
                else:  # using prediction as new value
                    positional_encodings_new_val = src[i + 1, :, 1:].unsqueeze(0)
                    predicted_humidity = torch.cat((prediction[-1, :, :].unsqueeze(0), positional_encodings_new_val),
                                                   dim=2)
                    sampled_src = torch.cat((sampled_src.detach(), predicted_humidity.detach()))

            """To update model after each sequence"""
            loss = criterion(target[:-1, :, 0].unsqueeze(-1), prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        if train_loss < min_train_loss or epoch % 10 == 0:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss if train_loss < min_train_loss else min_train_loss
            best_model = f"best_train_{epoch}.pth" if train_loss < min_train_loss else best_model

        if epoch % 10 == 0:  # Plot 1-Step Predictions

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')
            sampled_src_price = scaler.inverse_transform(sampled_src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
            src_price = scaler.inverse_transform(src[:, :, 0].cpu())  # torch.Size([35, 1, 7])
            target_price = scaler.inverse_transform(target[:, :, 0].cpu())  # torch.Size([35, 1, 7])
            prediction_humidity = scaler.inverse_transform(
                prediction[:, :, 0].detach().cpu().numpy())  # torch.Size([35, 1, 7])
            plot_training_3(epoch, path_to_save_predictions, src_price[:, 0], sampled_src_price[:, 0],
                            prediction_humidity[:, 0], index_in, index_tar)

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)

    plot_loss(path_to_save_loss, train=True)
    return best_model
