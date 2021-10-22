import numpy as np

from model import Transformer
import logging
import time  # debugging
from tqdm import tqdm
from plot import *
from helpers import *
from joblib import load


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):

    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    smape_list = []
    with torch.no_grad():

        model.eval()
        scaler = load('scalar_item.joblib')
        for index_in, index_tar, _input, target in tqdm(dataloader, 'BATCH'):
                
                src = _input.permute(1,0,2).double().to(device)  # L, B, F7: t1 -- tL
                target = target.permute(1,0,2).double().to(device)  # tL+1 - tL+W

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window):
                    
                    prediction = model(next_input_model, device)  # L,B,1: t2' - tL+1'

                    if all_predictions == []:
                        all_predictions = prediction  # L,B,1: t2' - tL+1'
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0)))  # L+,1N,1: t2' - tL+1', tL+2', tL+13'
                    if i != forecast_window-1:
                        pos_encoding_old_vals = src[1:, :, 1:] if i == 0 else pos_encodings[1:, :, :]
                        pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(0) # 1, B, 6,  append positional encoding of last predicted value: tL+1
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # L, B, 6 positional encodings matched with prediction: t2 -- tL+1

                        next_input_model = torch.cat((next_input_model[1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- tL, tL+1'
                        next_input_model = torch.cat((next_input_model, pos_encodings), dim=2) # L, B, 7 input for next round

                true = torch.cat((src[1:,:,0],target[:,:,0]))
                loss = criterion(true, all_predictions[:,:,0])
                val_loss += loss

                src_price = scaler.inverse_transform(src[:, :, 0].cpu())
                target_price = scaler.inverse_transform(target[:, :, 0].cpu())
                prediction_price = scaler.inverse_transform(all_predictions[:, :, 0].detach().cpu().numpy())
                for i in range(10):
                    plot_prediction(i, path_to_save_predictions, src_price[:, i], target_price[:, i],
                                    prediction_price[:, i], index_in, index_tar)
                    smape_list.append(smape(true[:,i].cpu().numpy(), all_predictions[:,i,0].cpu().numpy()))
                break

        val_loss = val_loss / len(dataloader)

        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")
        print(smape_list)
        print(np.mean(smape_list))