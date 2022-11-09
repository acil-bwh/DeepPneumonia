import os
import re
import h5py as f
import argparse
from tensorflow import keras

def predicciones_modelo(dataframes_path, model_name):
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False

    model = os.path.join('./models', model_name)
    model = keras.models.load_model(model)

    dataframes = f.File(dataframes_path, "r")
    for key in dataframes.keys():
        globals()[key] = dataframes[key]
    
    model_name = model_name[:-3]
    results = ev.evaluate(model, X_val, y_val, list(range(len(y_val))), mask=mask)
    ev.save_eval(model_name, 'validation', results)
    pred.save_metricas(model_name, 'testing', model, X_val, y_val, list(range(len(y_val))), mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    parser.add_argument('-mo',
                        '--model_name',
                        help="nombre del modelo",
                        type=str,
                        default='pneumonia_classification_model.h5')
    parser.add_argument('-h5',
                        '--h5_dataset',
                        type=str,
                        default='/home/mr1142/Documents/DeepPneumonia/data/training_validation_dataset.h5',
                        help="h5 dataset file with train and test folders")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_name = args.model_name
    import evaluation_functions.prediction as pred
    import evaluation_functions.evaluation as ev
    predicciones_modelo(args.h5_dataset, model_name)