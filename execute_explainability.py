import os
import re
import pickle
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import explainability.grad_cam as gc
import explainability.mask_quantification as msk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=3)
    parser.add_argument('-m',
                        '--model',
                        help="model name",
                        type=str,
                        default='pneumonia_classification_model')
    parser.add_argument('-im',
                        '--image',
                        help="images path over which explainability is going to be applied",
                        type=str,
                        default='./data/external_dataset/test')
    parser.add_argument('-th',
                        '--threshold',
                        help="heatmap threshold",
                        type=float,
                        default=0.1)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_path = os.path.join('./models',args.model + '.h5')
    images_path = args.image
    th = args.threshold

    # Check if the model uses mask
    if bool(re.search('mask', args.model)):
        mask = True
    else:
        mask = False

    # Load images
    images = [filename for filename in os.listdir(images_path) if 
                filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))][0:10]

    # Load mask model
    import image_functions.mask_model as mask_model
    proportions = []

    # Create dir where heatmaps and proportions are going to be saved
    save_dir = os.path.join('./results/heatmaps', args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_path in tqdm(images):
        # Grand cam image
        image = cv2.imread(os.path.join(images_path, image_path))
        model = tf.keras.models.load_model(model_path)
        grand_cam_im, heatmap = gc.apply_grandcam(model, mask, image)
        cv2.imwrite(os.path.join(save_dir, re.split('/', image_path)[-1]), np.array(grand_cam_im))

        # Heatmap inside mask
        mask_img = msk.apply_mask(image, mask_model.model)
        heatmap = cv2.resize(heatmap, (256, 256))
        binary_hm = (heatmap > th) *1
        proportions.append(msk.extract_proportion(binary_hm, mask_img))
    
    # Save proportions
    with open(os.path.join(save_dir, "proportions"), "wb") as fp:
         pickle.dump(proportions, fp)
