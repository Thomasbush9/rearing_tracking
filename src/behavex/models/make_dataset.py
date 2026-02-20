from argparse import ArgumentParser
from behavex.models.train_transformer import (
    prepare_and_save_dataset,
    prepare_and_save_dataset_chunked,
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data file or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output file or directory"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--window_size", type=int, default=128, help="Window size")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument(
        "--drop-cols",
        dest="drop_cols",
        type=str,
        nargs="+",
        default=[],
        help="Columns to exclude (use all if not set)",
    )
    parser.add_argument(
        "--keep-cols",
        dest="keep_cols",
        type=str,
        nargs="+",
        default=[],
        help="Use only these columns (overrides --drop-cols)",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Build dataset per-session and save as .h5 to avoid high memory",
    )
    args = parser.parse_args()
    common = dict(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        window_size=args.window_size,
        stride=args.stride,
        drop_cols=args.drop_cols if args.drop_cols else None,
        keep_cols=args.keep_cols if args.keep_cols else None,
        use_dist_head_event=True,
        trim_before_dist_head=False,
    )
    if args.chunked:
        prepare_and_save_dataset_chunked(args.data, args.output, **common)
    else:
        prepare_and_save_dataset(args.data, args.output, **common)


if __name__ == "__main__":
    main()
