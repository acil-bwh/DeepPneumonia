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
                        default='')
    parser.add_argument('-m',
                        '--model_name',
                        help="model to apply",
                        type=str,
                        default='pneumonia_classification_model')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = args.path
    model_name = args.model_name
    model_path = './models/'+ model_name + '.h5'

    print(model_name)

    # Check if the model uses mask
    if bool(re.search('mask', model_name)):
        mask = True
    else:
        mask = False
    
    # Check if the path exists and has images
    if os.path.exists(path) and len([im for im in os.listdir(path) if 
                            im.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]) > 0:
        # Load model
        model = keras.models.load_model(model_path)

        # Get images names and their prediction tensor
        images_names, prediction = ev.prediction_tensor(model, path, mask = mask)

        # Generate dataframe with images and prediction tensor and save it
        df = ev.results_dataframe(images_names, prediction)
        df.to_csv(os.path.join(path, model_name + '_results.csv'), index = False)
    else:
        print("\n Images path does not exist or it does not have any image!! \n Introduce a new path: python apply_model.py -p path")
