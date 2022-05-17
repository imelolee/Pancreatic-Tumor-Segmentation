import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--seed', type=int, default=2021, help='random seed')


# Train
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--train_dir', default='dataset/train', type=str) 
parser.add_argument('--valid_dir', default='dataset/val', type=str)
parser.add_argument('--checkpoint', type=str, default=None, help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--results_dir', default='./results', type=str)
parser.add_argument('--n_labels', type=int, default=2,help='number of classes')



args = parser.parse_args()