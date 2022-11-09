import os
import h5py as f
import other_functions.logs as logs
import evaluation_functions.evaluation as ev
import argparse
import numpy as np
import pickle
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf


def create_model(input_shape, backbone_name, frozen_backbone_prop):
    if backbone_name == 'IncResNet':
        backbone = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == 'EffNet3':
        backbone = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == 'Xception':
        backbone = Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(backbone)
    model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(layers.Dropout(0.2, name="dropout_out_1"))
    model.add(layers.Dense(768, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_2"))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dense(3, activation="softmax", name="fc_out"))

    # Set frozen proportion
    fine_tune_at = int(len(backbone.layers)*frozen_backbone_prop)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def generate_index(trainprop = 0.8):
    with open("./index/train", "rb") as fp:
        index = pickle.load(fp)

    np.random.shuffle(index)
    idtrain = index[:int(len(index)*trainprop)]
    idtest = index[int(len(index)*trainprop):]

    return idtrain, idtest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name to give to the model")
    parser.add_argument('-mo',
                        '--modelo',
                        type=str,
                        default='Xception',
                        help="which type of model")                      
    parser.add_argument('-f',
                        '--frozen_prop',
                        type=float,
                        default=0.4,
                        help="proportion of layers to frozen from backbone")
    parser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=8,
                        help="batch size")
    parser.add_argument('-lr',
                        '--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument('-m',
                        '--mask',
                        type=bool,
                        default=False,
                        help="apply mask")
    parser.add_argument('-h5',
                        '--h5_dataset',
                        type=str,
                        default='/home/mr1142/Documents/ACIL_data_repo/DeepPneumonia/data/training_validation_dataset.h5',
                        help="h5 dataset file with train and test folders")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    backbone = args.modelo
    frozen_prop = args.frozen_prop
    batch = args.batch
    lr = args.lr
    mask = args.mask
    trainprop = 0.8
    epoch = 100
    pix = 512

    # DATAFRAME
    df = f.File(args.h5_dataset, "r")
    for key in df.keys():
        globals()[key] = df[key]

    # DATA GENERATORS
    idtrain, idtest = generate_index(trainprop)

    from image_functions.data_generator import DataGenerator as gen
    traingen = gen(X_train, y_train, batch, pix, idtrain, mask)
    testgen = gen(X_train, y_train, batch, pix, idtest, mask)

    # MODEL
    input_shape = (pix,pix,3)
    model = create_model(input_shape, backbone, frozen_prop)    

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                    loss = 'categorical_crossentropy',
                    metrics = ['BinaryAccuracy', 'Precision', 'AUC'])

    # NAME
    if mask:
        name = name + '_mask_' + backbone + '_' + 'fine-0' + str(frozen_prop)[2:] + '_batch-' + str(batch) + '_lr-' + str(lr)[2:]
    else:
        name = name + '_' + backbone + '_' + 'fine-0' + str(frozen_prop)[2:] + '_batch-' + str(batch) + '_lr-' + str(lr)[2:]

    # CALLBACK
    callb = [logs.tensorboard(name), logs.early_stop(5)]

    # TRAIN
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)
    
    # SAVE METRICS
    import evaluation_functions.prediction as pred
    
    # Save train
    name = ev.save_training(history, name, 
            [backbone, frozen_prop, batch, lr, mask, trainprop, pix])
    print('TRAINING SAVED')

    # Save model
    model.save('./models/' + name + '.h5')
    print('MODEL SAVED')

    # Save evaluate and prediction
    idtest.sort()
    results = ev.evaluate(model, X_train, y_train, idtest, mask=mask)
    ev.save_eval(name, 'testing', results)
    print('EVALUATE SAVED')
    pred.save_metricas(name, 'testing', model, X_train, y_train, idtest, mask=mask)
    print('PREDICTION SAVED')
