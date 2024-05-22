import argparse


def argparse_option():
    parser = argparse.ArgumentParser('Arguments for OPP-PersonReID')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    # parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')

    # optimization
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

    # model dataset
    parser.add_argument('--dataset', type=str, default='uiuc',
                        choices=['scene', 'uiuc', 'mit', 'cifar', 'caltech'], help='dataset')
    parser.add_argument('--iteration', type=int, default=150, help='optimization iteration')
    parser.add_argument('--lp_iteration', type=int, default=400, help='optimization lp_iteration')
    parser.add_argument('--diffusion_iteration', type=int, default=50, help='optimization diffusion_iteration')

    # other setting
    parser.add_argument('--GDC', action='store_true', default=True, help='RECT loss function')
    parser.add_argument('--lamb', type=float, default=1.6, help='coefficient for the RECT loss function')
    parser.add_argument('--sigma', type=float, default=1, help='parameter of Gaussian Kernel')
    parser.add_argument('--sigmaGD', type=float, default=1, help='parameter of Gaussian Kernel in diffusion')
    parser.add_argument('--alpha', type=float, default=0.9, help='parameter of label propagation')
    parser.add_argument('--alphaGDC', type=float, default=0.9, help='parameter of Graph diffusion')
    parser.add_argument('--nearest', type=int, default=3, help='K nearest points')
    parser.add_argument('--k_nums', type=int, default=0,
                        help='K nearest points in diffusion if k==0 then full connection')
    parser.add_argument('--scale', action='store_true', default=True, help='K nearest points')
    parser.add_argument('--seed', default=10, type=int, help='for reproducibility')

    opt = parser.parse_args()

    return opt
