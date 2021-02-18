import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

from sklearn import datasets
from sklearn.decomposition import PCA

# see: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py

sys.path.append('../classic_detector')
from datasets.lunar_analogue import LunarAnalogueDataGenerator
from utils import tools



config_file = tools.handle_command_line_arguments()

with open(config_file) as f:
    config = yaml.full_load(f)

data_obj = LunarAnalogueDataGenerator(config)
gen = data_obj.create_generator('train')

print(gen)

batch_out = next(gen)
print(type(batch_out))
print(batch_out[0].shape)
print(np.max(batch_out[0]), np.min(batch_out[0]))
print(np.max(batch_out[1]), np.min(batch_out[1]))

B, H, W, C = batch_out.shape

batch_flat = batch_out.reshape(B, (H*W*C))
print(batch_flat.shape)
print(np.max(batch_flat[0]), np.min(batch_flat[0]))
print(np.max(batch_flat[1]), np.min(batch_flat[1]))
#%%

plt.imshow(tools.unstandardize_batch(batch_out[4]), interpolation='nearest')
plt.show()

## Consider using StandardScaler for sklearn research
pca_obj = PCA(n_components=2)
reduced_data = pca_obj.fit_transform(batch_flat)
recon = pca_obj.inverse_transform(reduced_data)

for i, comp in enumerate(pca_obj.components_):
    plt.imshow(tools.unstandardize_batch(comp.reshape(H, W, C)), interpolation='nearest')
    print(i)
    plt.show()

for i, (red_data, orig_data) in enumerate(zip(recon, batch_out)):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(tools.unstandardize_batch(red_data.reshape(H, W, C)), interpolation='nearest')
    ax[1].imshow(tools.unstandardize_batch(orig_data.reshape(H, W, C)), interpolation='nearest')
    print(i)
    plt.show()
    if i > 10:
        break
#%%

data = datasets.load_digits()

plt.imshow(data.images[0], cmap='gray', interpolation='nearest')
plt.show()

pca_obj = PCA()

reduced_data = pca_obj.fit_transform(data.data)
recon = pca_obj.inverse_transform(reduced_data)
print(recon.shape)

print(dir(pca_obj))
print(pca_obj.explained_variance_)
print(pca_obj.components_.shape)

for i, comp in enumerate(pca_obj.components_):
    plt.imshow(comp.reshape(8,8), cmap='gray', interpolation='nearest')
    print(i)
    plt.show()
    
for i, (red_data, orig_data) in enumerate(zip(recon, data.data)):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(red_data.reshape(8,8), cmap='gray', interpolation='nearest')
    ax[1].imshow(orig_data.reshape(8,8), cmap='gray', interpolation='nearest')
    print(i)
    plt.show()
    if i > 10:
        break

