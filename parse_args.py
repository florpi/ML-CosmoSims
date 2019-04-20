import argparse

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description="main.py")

    # Parameters for Dataset
    parser.add_argument(
            "--features_path",
            type=str,
            default='/cosma5/data/dp004/dc-beck3/Dark2Light/data/dark_matter_only/',
            help="path to datasets used to train,validate,test",
    )
    parser.add_argument(
            "--targets_path",
            type=str,
            default='/cosma5/data/dp004/dc-beck3/Dark2Light/data/full_physics/',
            help="path to datasets used to train,validate,test",
    )
    parser.add_argument(
            "--snapshot_nr",
            type=int,
            default=45,
            help="Snapshot number",
    )
    parser.add_argument(
            "--voxle_nr",
            type=int,
            default=1024,
            help="Nr. of elements in which simulation box is devided",
    )

    # Parameters for ML-algorithm
    parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    parser.add_argument("--print_freq", type=int, default=400, help="")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--model_idx",
        type=int,
        default=0,
        help="0: Unet 1: Baseline 2: Inception 3: R2Unet 4: two-phase model(classfication phase: one layer Conv, regression phase: R2Unet) 5: two-phase model(classfication phase: R2Unet, regression phase: R2Unet) 6: R2Unet attention 7: two-phase model(classfication phase: Inception, regression phase: R2Unet) 8: Incetion regression",
    )
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--loss_weight",
        type=float,
        default=20,
        help="weight of the loss equals to normalized [x, loss_weight * x,loss_weight * x]",
    )
    parser.add_argument(
        "--label_type",
        default="count",
        help="the label type we want to predict, count or mass",
    )
    parser.add_argument(
        "--target_class", type=int, default=0, help="0:classification 1:regression"
    )
    parser.add_argument("--load_model", type=int, default=0, help="")
    parser.add_argument(
        "--save_name",
        default="",
        help="the name of the saved model file, default don't save",
    )
    parser.add_argument(
        "--record_results",
        type=int,
        default=0,
        help="whether to write the best results to all_results.txt",
    )
    parser.add_argument(
        "--vel",
        type=int,
        default=0,
        help="whether to include velocity to the input (input dim 1 if not, 4 if yes)",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=0,
        help="whether to normalize the input(dark matter density)",
    )
    return parser.parse_args()

