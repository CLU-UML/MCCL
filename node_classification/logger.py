# -*- coding: utf-8 -*-
import torch
import datetime

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        
    def get_log_file(self,args):
        var = [
            args.dataset,
            args.num_layers,
            args.hidden_channels,
            args.dropout,
            args.lr,
            args.l2,
            args.epochs,
            args.runs,
            args.feat,
            args.prioritizing_approach ,
            args.metric_order,
            args.loss_creteria,
            args.add_random,
            str(datetime.datetime.now()).replace(" ","_")
            
        ]
        
        log_filename  = "/DIR_LOCATION_TO_WRITE_LOGS/logs"+ "/{}_layers_{}_{}_dropout_{}_lr_{}_l2_{}_e_{}_runs_{}_feat_{}prio_approach_{}_metric_order_{}_loss_creteria_{}_add_random_{}_{}".format(*var)
        
        return log_filename

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
