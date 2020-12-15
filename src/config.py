import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='command for train pyramid model')

    parser.add_argument('--GPUID', type=str, default='0', help='gpu ids')

    parser.add_argument('--root', type=str,
                        default='data/Market-1501-v15.09.15')

    parser.add_argument('--batch_id', type=int, default=8)
    parser.add_argument('--batch_image', type=int, default=8)
    parser.add_argument('--batch_train', type=int, default=64)
    parser.add_argument('--batch_test', type=int, default=64)

    parser.add_argument('--trple_margin', type=float, default=1.4)
    parser.add_argument('--para_balance', type=float, default=0.0)

    parser.add_argument('--transform_imsize', type=list, default=[384, 128])
    parser.add_argument('--transform_norm_mean', type=list,
                        default=[0.485, 0.456, 0.406])
    parser.add_argument('--transform_norm_std', type=list,
                        default=[0.229, 0.224, 0.225])
    parser.add_argument('--transform_random_erase_p', type=float, default=0.2)
    parser.add_argument('--transform_random_erase_mean',
                        type=list, default=[0.0, 0.0, 0.0])

    parser.add_argument('--lr_finetune', type=float, default=0.01)
    parser.add_argument('--lr_new', type=float, default=0.1)
    parser.add_argument('--lr_schedule', type=list,
                        default=[60, 70, 80, 90, 100])
    parser.add_argument('--n_epoch', type=int, default=120)

    parser.add_argument('--data_loader', type=str, default='Market1501')

    args = parser.parse_args()

    return args
