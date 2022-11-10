import os
import argparse
import json
import numpy as np
from mango import Tuner, scheduler
from scipy.stats import uniform


# PARAM SPACE
param_space = dict(backbone=['Xception', 'IncResNet', 'EffNet3'],
                    frozen_prop = uniform(0,1),
                    lr= uniform(1e-5, 1e-3),
                    mask = [True, False])


# Configuration
conf_dict = dict(num_iteration=50)


# EXECUTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-h5',
                        '--h5_dataset',
                        type=str,
                        default='./data/training_validation_dataset.h5',
                        help="h5 dataset file with train and test folders")
    parser.add_argument('-ev',
                        '--evaluation_type',
                        type=str,
                        default='internal',
                        help="evaluation over internal or external dataset")
    parser.add_argument('-ex',
                        '--external_df',
                        type=str,
                        default='./data/external_dataset/test',
                        help="external dataset path for validation (should be .csv)")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    import other_functions.hyperparameter_trainer as tr

    # OBJETIVE
    # f1 score of tree trainings with the same model
    @scheduler.serial
    def objective(**params):
        print('--------NEW COMBINATION--------')
        print(params)
        results = []
        for x in range(3):
            results.append(tr.train(**params, 
                                    dataframe_path=args.h5_dataset,
                                    evaluation_type=args.evaluation_type,
                                    external_dataframe_path=args.external_df))
            print('results {}: {}'.format(x, results[x]))
        print('FINAL RESULTS {}'.format(np.mean(results)))
        return np.mean(results)

    # Generate tuner and maximize
    tuner = Tuner(param_space, objective, conf_dict)
    results = tuner.maximize()

    # Save resuls in a json
    for k, v in results.items():
        if type(v) is np.ndarray:
            results[k] = list(v)

    with open('./results/hyperparameter_tuning/results_' + args.evaluation_type + '.json', 'w') as j:
        json.dump(results, j)

    # Print best results
    print('best parameters:', results['best_params'])
    print('best f1score:', results['best_objective'])


