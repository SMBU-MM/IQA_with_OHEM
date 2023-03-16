import torch
import torchvision
from BaseCNN import BaseCNN
from PIL import Image
import numpy as np
import os
import argparse

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


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def eval(test_file, test_file_path, save_file, idx=1):
    config = parse_config()
    model = torch.nn.DataParallel(BaseCNN(config).cuda())
    # load our pre-trained model on the koniq-10k dataset
    model.load_state_dict(torch.load('UNIQUE/{}/checkpoint/DataParallel-00007.pt'.format(idx))['state_dict'])
    model.eval()
    
    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])
    
    # random crop 10 patches and calculate mean quality score
    with open(os.path.join(test_file_path, test_file), 'r') as rfile:
        lines = rfile.readlines()  # 使用csv.reader读取csvfile中的文件
    count = 0
    with open(save_file, 'w') as wfile:
        for line in lines:  # 将csv 文件中的数据保存到birth_data中
            line_list = line.replace('\n','').split('\t')
            img_name = line_list[0]
            img_path = os.path.join(test_file_path, img_name)
            mos_value = float(line_list[1])
            img = pil_loader(img_path)
            img = transforms(img)
            img = torch.tensor(img.cuda()).unsqueeze(0)
            pred, _ = model(img)  # 'paras' contains the network weights conveyed to target network
    
            wfile.write("{},{},{}\n".format(img_name, mos_value, pred.item()))
            count += 1
            if count % 10 == 0:
                print(count)

if __name__ == '__main__':
    for idx in range(1, 11):
        test_file = 'kadid10k_splits3/{}/kadid10k_test.txt'.format(idx)
        test_file_path = '../kadid10k/'
        save_file = './preds/kadid10k_test_split{}.txt'.format(idx)
        eval(test_file, test_file_path, save_file, idx) 
        