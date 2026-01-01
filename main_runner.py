import argparse
import warnings

# Suppress the specific warning from torchvision/io/image.py
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io.image"
)
import sys
import os
sys.path.append(os.path.abspath('.'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--attack', dest='attack', required=True,
                        choices=['dba', 'cas', 'cerp', 'badnet','edgecase'],
                        help='Type of attack to run')
    parser.add_argument('--defense', dest='defense', required=True, choices=['mean', 'pca-deflect', 'nab', 'npd'], help='Type of defense to run')
    parser.add_argument('--dataset', dest='dataset', choices=['cifar','mnist','fmnist','emnist'])
    args = parser.parse_args()
    if args.attack == 'dba':
        import attacks.DBA.main as main
        main.main_dba(args.defense, args.dataset)
    elif args.attack == 'cas':
        import attacks.ConstrainScale.main as main
        main.main_cas(args.defense, args.dataset)
    elif args.attack == 'cerp':
        import attacks.CerP.main as main
        main.main_cerp(args.defense, args.dataset)
    elif args.attack == 'edgecase':
        import attacks.Edgecase.main as main
        main.main_edge(args.defense, args.dataset)
    elif args.attack == 'badnet':
        import attacks.Badnet.main as main
        main.main_badnet(args.defense, args.dataset)