import argparse
import TrainModel
import os
import random
import numpy as np

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=str, default='0')
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument("--fz", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--loss", type=str, default='ssl') # naive | joint | joint_div | ssl
    parser.add_argument("--unlabeled", type=str, default='koniq-10k') # koniq-10k | spaq 
    parser.add_argument("--train_txt", type=str, default="train.txt") #  
    parser.add_argument("--num_per_round", type=int, default=1800) # the number of images selected in each round for active learning 
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--num_pl_pairs", type=int, default=40000) # the number of images selected in each round for active learning 
    parser.add_argument("--cls_fn", type=str, default='sce0.5') # the number of images selected in each round for active learning 

    parser.add_argument("--split", type=int, default=8)
    parser.add_argument("--path_idx", type=int, default=8)
    parser.add_argument("--trainset", type=str, default="../databaserelease2/")
    parser.add_argument("--live_set", type=str, default="../databaserelease2/")
    parser.add_argument("--bid_set", type=str, default="../BID/")
    parser.add_argument("--kadid10k_set", type=str, default="../kadid10k/")
    parser.add_argument("--clive_set", type=str, default="../ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="../koniq-10k/")

    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--batch_size2", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--num_round", type=int, default=1)
    parser.add_argument("--start_round", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)
    return parser.parse_args()


def main(cfg):
    t = TrainModel.Trainer(cfg)
    t.fit()
    
if __name__ == "__main__":
    config = parse_config()
    # set the seed of random 
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_idx
    print("[*] gpu index: {}".format(config.gpu_idx))
    config.ckpt_path = 'checkpoints'
    config.ckpt_best_path = 'bests'
    
    for i in range(1,11):
        config.split = i
        #########################################################################################################################
        # create file folder
        #########################################################################################################################
        config.trainset_path = '../databaserelease2/splits2/{}/'.format(i)
        config.train_file = os.path.join(config.trainset_path, 'live_train.txt') 
        config.test_path = '../kadid10k/splits9/{}/'
        config.valid_file1 =  os.path.join(config.test_path, 'kadid10k_valid_score.txt')
        config.test_file1 =  os.path.join(config.trainset_path, 'live_test.txt')
        config.test_file2 =  os.path.join(config.test_path, 'kadid10k_test_score.txt')
        #############################################################################################################################
        # retrain the model from scratch
        #############################################################################################################################
        #stage1: freezing previous layers, training fc in the first round
        config.fz = True
        config.resume = False 
        config.max_epochs = 3
        config.batch_size = 96
        main(config) 

        # stage2: fine-tuning the whole network
        config.fz = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = 12
        config.batch_size = 16
        main(config)