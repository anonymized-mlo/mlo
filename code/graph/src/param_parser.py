import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GNN.")
    parser.add_argument("--num-runs",
                        nargs="?",
                        default=1,
	                help="Num runs, default 1")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/tr_edge_hr_att_0120.csv",
	                help="Edge list csv.")
    parser.add_argument("--edge-fe-path",
                        nargs="?",
                        default="./input/tr_edge_fe_att_0120.csv",
	                help="Edge features.")
    parser.add_argument("--features-path",
                        nargs="?",
                        default="./input/tr_node_fe_att_0120.csv",
	                help="Edge list csv.")
    parser.add_argument("--embedding-path",
                        nargs="?",
                        default="./output/embedding/lux_gnn.csv",
	                help="Target embedding csv.")
    parser.add_argument("--regression-weights-path",
                        nargs="?",
                        default="./output/weights/lux_gnn.csv",
	                help="Regression weights csv.")
    parser.add_argument("--fc-weights-path",
                        nargs="?",
                        default="./output/fc_weights/lux_gnn.csv",
	                help="FC weights csv.")
    parser.add_argument("--log-path",
                        nargs="?",
                        default="./logs/lux_logs.json",
	                help="Log json.")
    parser.add_argument("--epochs",
                        type=int,
                        default=400,
	                help="Default 400.")
    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
	                help="Default 30.")
    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64,
	                help="Default64.")
    parser.add_argument("--seed",
                        type=int,
                        default=33,
	                help="Default 33.")
    parser.add_argument("--lamb",
                        type=float,
                        default=1.0,
	                help="Default 1.0.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.15,
	                help="Default 0.15.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Default 0.01.")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5,
	                help="Default 10^-5.")
    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help=" e.g. 32 32.")
    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_false")
    parser.add_argument("--edge-features-include",
                        dest="edge_features_incls",
                        action = "store_true",
                        help = "Whether to include edge features in the final classification")
    parser.add_argument("--edge-features-exclude",
                        dest="edge_features_incls",
                        action = "store_false",
                        help = "Whether to exclude edge features in the final classification")
    parser.add_argument("--attention-include",
                        dest="attention_include",
                        action = "store_true",
                        help = "Whether to include attention")
    parser.add_argument("--attention-exclude",
                        dest="attention_include",
                        action = "store_false",
                        help = "Whether to exclude attention")

    parser.set_defaults(spectral_features=False)
    parser.set_defaults(attention_include=True)
    parser.set_defaults(edge_features_incls=True)
    parser.set_defaults(layers=[32, 32])
    return parser.parse_args()
