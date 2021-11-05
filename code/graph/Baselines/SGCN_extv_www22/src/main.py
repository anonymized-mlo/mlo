"""SGCN runner."""

from sgcn_updated import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs
import numpy as np

def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    avg_auc = []
    avg_f1 = []
    avg_precision = []
    avg_recall = []
    avg_acc = []

    for x in range(int(args.num_runs)):
        print("Iteration: ", x)
        tab_printer(args)
        edges = read_graph(args)
        trainer = SignedGCNTrainer(args, edges)
        trainer.setup_dataset()
        trainer.create_and_train_model()
        if args.test_size > 0:
            trainer.save_model()
            score_printer(trainer.logs)
            save_logs(args, trainer.logs)
            avg_auc.append(score_printer(trainer.logs, avg='auc')[0])
            print("This run's AUC: ", "%.3f" % (score_printer(trainer.logs, avg='auc')[0]))
            print('-----')
            avg_f1.append(score_printer(trainer.logs, avg='auc')[1])
            avg_precision.append(score_printer(trainer.logs, avg='auc')[2])
            avg_recall.append(score_printer(trainer.logs, avg='auc')[3])
            avg_acc.append(score_printer(trainer.logs, avg='auc')[4])

    print('AUC averaged over {} runs: '.format(args.num_runs), "%.3f" % np.mean(avg_auc))
    print('F1 averaged over {} runs: '.format(args.num_runs), "%.3f" % np.mean(avg_f1))
    print('Precision averaged over {} runs: '.format(args.num_runs), "%.3f" % np.mean(avg_precision))
    print('Recall averaged over {} runs: '.format(args.num_runs), "%.3f" % np.mean(avg_recall))
    print('Accuracy averaged over {} runs: '.format(args.num_runs), "%.3f" % np.mean(avg_acc))
    print('Max AUC: ', "%.3f" % max(avg_auc), 'Max F1: ', "%.3f" % max(avg_f1), 'Max Precision: ', "%.3f" % max(avg_precision), \
    'Max Recall: ', "%.3f" % max(avg_recall), 'Max Accuracy', "%.3f" % max(avg_acc))
if __name__ == "__main__":
    main()
