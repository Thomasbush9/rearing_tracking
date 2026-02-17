from argparse import ArgumentParser
from behavex.models.train_transformer import prepare_and_save_dataset

def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data file or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output file or directory")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--window_size", type=int, default=128, help="Window size")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--drop_cols", type=str, nargs="+", default=[], help="Columns to drop")
    args = parser.parse_args()
    prepare_and_save_dataset(
        args.data,
        args.output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        drop_cols=args.drop_cols if args.drop_cols else None,
    )

if __name__ == "__main__":
    main()