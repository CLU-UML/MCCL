import os
import warnings
warnings.filterwarnings('ignore')

import log
from train import train_model_gtnn, train_model_w_competence, init_model
from  evaluate import  eval_best_model
from torch_geometric.data import DataLoader

import variables
import argparse
import utils

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    import time
    start = time.time()
    
    parser = argparse.ArgumentParser('Interface for GTNN framework')
    parser.register('type', bool, utils.str2bool)  # add type keyword to registries

    parser.add_argument('--dataset', type=str, default='pgr', help='dataset name - pgr, gdpr')

    # model hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 regularization weight')
    
    parser.add_argument('--curriculum_length', type=int, default=100, help="length of curriculum")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--device', type=str, default="cuda:0", help="gpu-device")

    # model configuration
  
    #parser.add_argument('--training_type', type=str, default="regular", help="type of training as regular, curriculum (sl), curriculum with trend (sl_trend) ")
    parser.add_argument('--seed', type=int, default=925, help="seed")
    #parser.add_argument('--model_type', type=str, default="sage", help="model type")

    # dataset arguments
    parser.add_argument('--add_additional_feature', type=bool, default=True, help="to add or not to add additional feature")
    parser.add_argument('--prioritizing_approach', type=str, default="none", help="none, loss_based , complexity_score_based") # none for GTNN
    
    # metric order and loss creteria
    parser.add_argument('--metric_order', type=str, default= "A", help= " choose for ascending order:A/D")
    parser.add_argument('--loss_creteria', type=str, default= 'max', help= "loss_creteria: min/max")
    parser.add_argument('--use_k_means', type=bool, default= True, help= "T/F")
    parser.add_argument('--add_random', type=bool, default= False, help= "T/F")
    parser.add_argument('--evaluate_test_per_epoch', type=bool, default= False, help= "T/F to calculate the results for test (if True) after every epoch")
    

    args = parser.parse_args()
    utils.fix_seed(args.seed)

    train_set, val_set, test_set = utils.load_datasets(args)
    
    model_loc = "{}/{}_best.pth".format(variables.dir_model, args.dataset)
    print(model_loc)
    
    if False and os.path.exists(model_loc): # to load best model
        log_filename = log.create_log(args)
        print(args)
        print('Loading already trained model')
        model = init_model(args)
    else:
        log_filename = log.create_log(args)
        print(args)
        if args.prioritizing_approach == "none":
            if args.evaluate_test_per_epoch:
                model = train_model_gtnn(train_set, test_set,  args)
            else:
                model = train_model_gtnn(train_set, val_set,  args)
        else:
            if args.evaluate_test_per_epoch:
                model = train_model_w_competence(train_set, test_set,  args)
            else:
                model = train_model_w_competence(train_set, val_set,  args)
            
    test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    val_data_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print("for test:")
    p, r, f1, predictions, predictions_proba = eval_best_model(args, model, test_data_loader)
    prediction_file = log_filename.replace(".txt", "_test_prediction.txt")
    prediction_writer = open(prediction_file, "w")
    for pred,proba in zip(predictions, predictions_proba):
        prediction_writer.write("{}\t{}".format(pred.item(), proba.item()))
        prediction_writer.write("\n")
        prediction_writer.flush()

    print("for train:")
    p, r, f1, predictions, predictions_proba = eval_best_model(args, model, train_data_loader)
    prediction_file = log_filename.replace(".txt", "_train_prediction.txt")
    prediction_writer = open(prediction_file, "w")
    for pred,proba in zip(predictions, predictions_proba):
        prediction_writer.write("{}\t{}".format(pred.item(), proba.item()))
        prediction_writer.write("\n")
        prediction_writer.flush()

    print("for val:")
    p, r, f1, predictions, predictions_proba = eval_best_model(args, model, val_data_loader)
    prediction_file = log_filename.replace(".txt", "_val_prediction.txt")
    prediction_writer = open(prediction_file, "w")
    for pred,proba in zip(predictions, predictions_proba):
        prediction_writer.write("{}\t{}".format(pred.item(), proba.item()))
        prediction_writer.write("\n")
        prediction_writer.flush()



        
    done = time.time()
    elapsed = done - start
    print(elapsed)

if __name__ == "__main__":
    main()
