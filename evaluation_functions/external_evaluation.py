import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
import evaluation_functions.metrics_and_plots as met
import image_functions.prepare_img_fun as fu


def img_prepare(img, mask = False, pix = 512):
    try:
        img = fu.get_prepared_img(img, pix, mask, clahe_bool=True)
    except:
        print('random img')
        img = np.random.randint(0,255,512*512).reshape((512,512, 1))
    return img[np.newaxis,:]


def prediction_tensor(model, path, mask = False, pix = 512, batch_size = 80):
    images_names = os.listdir(path)
    images_names = [image for image in images_names 
                    if image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    batches = int(len(images_names)/batch_size)+1
    y_pred = []
    for batch in tqdm(range(batches)):
        batch_names = images_names[batch*batch_size:(batch+1)*batch_size]
        images = list(map(lambda x: img_prepare(cv2.imread(os.path.join(path, x)),
                                    mask, pix), batch_names))
        images = np.concatenate(images)
        y_pred.append(model.predict(images, verbose=0, batch_size=batch_size))
    y_pred = np.concatenate(y_pred)
    return images_names, y_pred


def results_dataframe(images, results):
    df = pd.DataFrame()
    df['name'] = images
    df['normal'] = results[:,0]
    df['mild'] = results[:,1]
    df['severe'] = results[:,2]
    return df


def binarize(array, threshold):
    array[array >= threshold] = 1
    array[array < threshold] = 0
    return array


def metricas_dict(real, pred):
    metrics_dict = {}
    plot_dict = {}
    metricas, plots = met.metrics_per_class('', real, pred[:,0])
    metrics_dict.update(metricas)
    plot_dict.update(plots)
    thresholds = ['younden_','pr_max_','pr_cut_']

    for threshold in thresholds:
        binar = binarize(pred[:,0].copy(), metricas[threshold])
        metrics_dict['f1_score_' + threshold] = metrics.f1_score(real, binar, 
                                                                average = 'weighted')
        metrics_dict['precision_score_' + threshold] = metrics.precision_score(real, 
                                                                            binar, 
                                                                            average = 'weighted')
        metrics_dict['recall_score_' + threshold] = metrics.recall_score(real, 
                                                                        binar, 
                                                                        average = 'weighted')
        metrics_dict['accuracy_score_' + threshold] = metrics.accuracy_score(real, binar)
    
    binar = binarize(pred[:,0].copy(), 0.5)
    metrics_dict['f1_score_' + str(0.5)] = metrics.f1_score(real, binar, 
                                                            average = 'weighted')
    metrics_dict['precision_score_' + str(0.5)] = metrics.precision_score(real, 
                                                                        binar, 
                                                                        average = 'weighted')
    metrics_dict['recall_score_' + str(0.5)] = metrics.recall_score(real, 
                                                                    binar, 
                                                                    average = 'weighted')
    metrics_dict['accuracy_score_' + str(0.5)] = metrics.accuracy_score(real, binar)
    
    binar = met.extract_max(pred.copy())[:,0]
    metrics_dict['f1_score_' + 'max'] = metrics.f1_score(real, binar, 
                                                            average = 'weighted')
    metrics_dict['precision_score_' + 'max'] = metrics.precision_score(real, 
                                                                        binar, 
                                                                        average = 'weighted')
    metrics_dict['recall_score_' + 'max'] = metrics.recall_score(real, 
                                                                    binar, 
                                                                    average = 'weighted')
    metrics_dict['accuracy_score_' + 'max'] = metrics.accuracy_score(real, binar)

    return metrics_dict, plot_dict


def execute_metrics(df):
    true = np.array(df.real)
    pred = np.array(df[['normal', 'mild', 'severe']])
    return metricas_dict(true, pred)


def calculate_metrics(df, path):
    df = df.sort_values('name').reset_index(drop = True)
    real = pd.read_csv(os.path.join(path, 'data.csv')).sort_values('img_name').reset_index(drop = True)
    real['real'] = real.normal
    real = real.drop('normal', axis = 1)
    df = pd.concat([real, df], axis = 1)
    df = df.dropna(axis = 0)
    return execute_metrics(df)


def save_in_csv(val_test, name, results):
    comparation = pd.read_csv(os.path.join('./results/external_validation/results_comparation_' +  val_test + '.csv'))
    comparation.loc[len(comparation)] = [name] + list(results[0].values())
    comparation.to_csv(os.path.join('./results/external_validation/results_comparation_' +  val_test + '.csv'), index = False)


def save_plots_fun(results, name):
    p = './results/external_validation'
    path = os.path.join(p, name)
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")
    for k, v in results[1].items():
        met.save_plot(v, path, k)