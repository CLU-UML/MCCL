import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Models are saved under "saved_models"
Logs are written in "logs" directory

'''
dir_model = "/DIR_LOCATION_TO_SAVE_MODEL/saved_models"
dir_logs = "/DIR_LOCATION_TO_WRITE_LOGS/logs"
dir_data = "/DATA_DIR"
