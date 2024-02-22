import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('--z', type=int, default=32, help='Latent dimension size')
    parser.add_argument('--dataset', type=int, default=3, help='Database to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='Fraction of training data to use per epoch')
    parser.add_argument('--run_name', type=str, default='', help='Run name, eg run-20240222_150106-9w4ls8zw')
    return parser.parse_args()
