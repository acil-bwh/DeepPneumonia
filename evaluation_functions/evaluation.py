import os
import pandas as pd


def save_training(history, name, otros_datos):
    datos = history.history
    name = name + '_auc-' + str(max(datos['val_auc']))[2:4]
    path = './results/train/train_max.csv'
    save_train_in_table(datos, name, otros_datos, path)
    path = './results/train/each_model_train'   
    pd.DataFrame(datos).to_csv(os.path.join(path, name + '_data.csv'), index = False)
    return name


def save_train_in_table(datos, name, otros_datos, path):
    df = pd.read_csv(path)
    values = [name]
    values.extend(otros_datos)
    for v in datos.values():
        values.append(max(v))
    df.loc[len(df)] = values
    df.reset_index(drop = True)
    df.to_csv(path, index = False)


def evaluate(model, X_val, y_val, index, batch = 8, pix = 512, mask = False):
    from image_functions.data_generator import DataGenerator as gen
    generator = gen(X_val, y_val, batch, pix, index, mask)
    results = model.evaluate(generator, batch_size=batch)
    print(results)
    return results


def save_eval(name, test_val, results):
    path = './results/' + test_val + '/evaluation.csv'
    df = pd.read_csv(path)
    save = [name] + results
    try:
        # Si ya existe el modelo, se sobreescriben las métricas
        i = df[df['nombre'] == name].index
        df.loc[i[0]] = save
    except:
        df.loc[len(df.index)] = save
    df.reset_index(drop=True)
    df.to_csv(path, index = False)
