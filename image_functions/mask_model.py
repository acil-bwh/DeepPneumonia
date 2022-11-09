import os
import tensorflow.keras as keras
import image_functions.losses as ex


model = os.path.join('./models', 'thorax_segmentation_model.h5')
model = keras.models.load_model(model, 
                                    custom_objects={"loss_mask": keras.losses.BinaryCrossentropy, 
                                                    "dice_coef_loss": ex.dice_coef_loss,
                                                    "dice_coef": ex.dice_coef})

print('\n\n MASK MODEL LOADED \n\n')