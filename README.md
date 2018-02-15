# DECO-Convnet
[DECO](https://wipac.wisc.edu/deco/home) computer vision model implemented in Python using the [Keras API](https://keras.io/).

## Example Usage
Train model with k-fold cross validation and data augmentation:
```python
import numpy as np
import h5py
import convnet

f = h5py.File('DECO_Image_Database.h5','r')
images = np.array(f['train/train_images'])
labels = np.array(f['train/train_labels'])

cnn = convnet.cnn(training=True)
cnn.train_with_kfold(images, labels, k_folds=10, seed=None, shuffle=True,
                     batch_size=32, epochs=10, initial_epoch=0, smooth_factor=None,
                     check_point=True, check_point_weights_only=True, horizontal_flip=True,
                     vertical_flip=True, width_shift_range=0.08, height_shift_range=0.08,
                     rotation_range=180., zoom_range=[0.9,1.1], fill_mode="constant", cval=0,
                     save_model=None, save_weights=None, save_history=None, output_dir=None)
              
f.close()
```
