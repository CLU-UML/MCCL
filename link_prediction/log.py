import sys
import variables
import datetime


def create_log(args):
    dataset = args.dataset
    lr = args.lr
    L2 = args.l2
    add_additional_feature = args.add_additional_feature
    curriculum_length = args.curriculum_length
    seed = args.seed
    approach = args.prioritizing_approach
    lc = args.loss_creteria
    mo = args.metric_order
    km = args.use_k_means
    random = args.add_random
    eval_test = args.evaluate_test_per_epoch
    time_stamp = str(datetime.datetime.now())
    log_var_order = [
        dataset,
        lr,
        L2,
        add_additional_feature,
        curriculum_length,
        seed,
        approach,
        lc,
        mo,
        km,
        random,
        eval_test,
        time_stamp,
        
    ]

    log_filename = variables.dir_logs + "/{}_{}_{}_addF_{}_cl_{}_s_{}_app_{}_{}_{}_km_{}_r_{}_eval_{}_{}.txt".format(
        *log_var_order)

    sys.stdout = sys.stderr = open(log_filename, 'w')
    sys.stdout.flush()
    
    return log_filename


