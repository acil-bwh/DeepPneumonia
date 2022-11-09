import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import image_functions.prepare_img_fun as fu
import evaluation_functions.metrics_and_plots as met


def img_predict(model, img, mask = False, pix = 512):
    try:
        img = fu.get_prepared_img(img, pix, mask)
    except:
        img = np.random.randint(0,255,512*512).reshape((512,512, 1))
    return model.predict(img[np.newaxis,:], verbose=0)


def prediction_tensor_old(model, X, index, mask = False, pix = 512):
    y_pred = np.zeros((len(index), 3))
    print('Prediction progress')
    for i in tqdm(range(y_pred.shape[0])):
        y_pred[i,...] = img_predict(model, X[index[i]], mask, pix)
    return y_pred


def img_prepare(img, mask = False, pix = 512):
    try:
        img = fu.get_prepared_img(img, pix, mask)
    except:
        img = np.random.randint(0,255,512*512).reshape((512,512, 1))
    return img[np.newaxis,:]


def prediction_tensor(model, X, index, mask = False, pix = 512, batch_size = 80):
    batches = int(len(index)/batch_size)+1
    y_pred = []
    for batch in tqdm(range(batches)):
        batch_index = index[batch*batch_size:(batch+1)*batch_size]
        images = list(map(lambda x: img_prepare(X[x],mask, pix), batch_index))
        images = np.concatenate(images)
        y_pred.append(model.predict(images, verbose=0, batch_size=batch_size))
    y_pred = np.concatenate(y_pred)
    return y_pred


def save_json(path, data):
    with open(os.path.join(path, 'metrics.json'), 'w') as j:
        json.dump(data, j)


def save_in_csv(path, name, metricas):
    file = 'prediction.csv'
    df = pd.read_csv(os.path.join(path, file))
    save = [name] + list(metricas.values())
    try:
        # If the model already exists metrics will be overwrited
        i = df[df['name'] == name].index
        df.loc[i[0]] = save
    except:
        df.loc[len(df.index)] = save
    df.reset_index(drop=True)
    df.to_csv(os.path.join(path, file), index = False)


def save_metricas(name, val_test, model, X, y, index, mask = False):
    y_pred = prediction_tensor(model, X, index, mask)
    y_real = y[index]
    print('predicton done')
    metricas, plots = met.metricas_dict(y_real, y_pred)
    print('metrics done')
    p = './results/' + val_test
    path = os.path.join(p, name)
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")
    try:
        save_json(path, metricas)
        print('json saved')
    except:
        print(metricas)
        print('json no saved')
    save_in_csv(p, name, metricas)
    print('saved in csv')
    for k, v in plots.items():
        met.save_plot(v, path, k)
    print('plots saved')
    met.class_report(y_real, y_pred, path)

