import argparse

def get_args():
    # general arguments for all methods
    parser = argparse.ArgumentParser()

    # other parameters
    parser.add_argument('--mode', type=str, default='UpperBound', 
                        help='UpperBound or Fed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # federated setup parameters
    parser.add_argument('--model', type=str, default='ResNet18',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo',
                        help='the data partitioning strategy')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='concentration parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--C', type=int, default=10,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='how many clients are sampled in each round')
    parser.add_argument('--approach', type=str, default='fedavg',
                        help='federated learning algorithm being used')
    parser.add_argument('--n_comm_round', type=int, default=1,
                        help='number of maximum communication rounds')
    parser.add_argument('--init_seed', type=int, default=42,
                        help="Random seed")

    # local training parameters
    parser.add_argument('--train_bs', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_bs', type=int, default=64,
                        help='batch size for testing')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of sgd optimizer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of local epochs')   
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay during local training")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='the optimizer')
    
    # logging parameters
    parser.add_argument('--print_interval', type=int, default=50,
                        help='how many comm round to print results on screen')
    parser.add_argument('--datadir', type=str, required=False, default="./data/",
                        help="Data directory")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/",
                        help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None,
                        help='The log file name')

    parser.add_argument('--ckptdir', type=str, required=False, default="./models/",
                        help='directory to save model')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='how many rounds do we save the checkpoint one time')

    args, appr_args = parser.parse_known_args()
    return args, appr_args