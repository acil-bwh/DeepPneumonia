import os
import re
import argparse
from tensorflow import keras
import evaluation_functions.external_evaluation as ev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-p',
                        '--path',
                        help="images path",
                        type=str,
                        default='./data/external_dataset')
    parser.add_argument('-vt',
                        '--val_test',
                        help="apply over test or validation dataset",
                        type=str,
                        default='val')
    parser.add_argument('-m',
                        '--model_name',
                        help="model to apply",
                        type=str,
                        default='pneumonia_classification_model')
    parser.add_argument('-sp',
                        '--save_plots',
                        help="save results plots",
                        type=bool,
                        default=False)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    val_test = args.val_test
    path = os.path.join(args.path, val_test)
    save_plots = args.save_plots
    model_name = args.model_name

    print(model_name)

    model_path = './models/'+ model_name + '.h5'
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False
    
    model = keras.models.load_model(model_path)

    images_names, prediction = ev.prediction_tensor(model, path, mask = mask)

    df = ev.results_dataframe(images_names, prediction)
    df.to_csv(os.path.join('./results/external_validation/model_results', model_name + '_' + val_test + '_results.csv'), index = False)

    results = ev.calculate_metrics(df, path)
    ev.save_in_csv(val_test, model_name, results)

    if save_plots:
        ev.save_plots_fun(results, model_name + '_' + val_test)
            
