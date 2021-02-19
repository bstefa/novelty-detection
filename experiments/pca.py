import numpy as np
import matplotlib.pyplot as plt
import yaml

from sklearn import datasets
from sklearn.decomposition import PCA
# see: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
from utils import tools
from datasets.lunar_analogue import LunarAnalogueDataGenerator
from models.incremental_pca import IncrementalPCA
from modules.pca_base_module import PCABaseModule

DEFAULT_CONFIG_FILE = 'configs/pca.yaml'
config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

# Instantiate a data generator class
datagenerator = LunarAnalogueDataGenerator(config)

model = IncrementalPCA()

module = PCABaseModule(datagenerator, model, config)

module.fit_pca()
module.transform_pca()

# # Grab statistics on the batch
# batch_in = next(gen)
# bs = tools.BatchStatistics(batch_in)
# print(bs.mean, bs.min, bs.max, bs.std)
# B, H, W, C = bs.shape
#
# # Flatten the batch for processing with PCA
# batch_flat = batch_in.reshape(B, (H*W*C))
#
# plt.imshow(tools.unstandardize_batch(batch_in[4]), interpolation='nearest')
# plt.show()
#
# ## Consider using StandardScaler for sklearn research
# pca_obj = PCA(n_components=2)
# reduced_data = pca_obj.fit_transform(batch_flat)
# recon = pca_obj.inverse_transform(reduced_data)
#
# for i, comp in enumerate(pca_obj.components_):
#     plt.imshow(tools.unstandardize_batch(comp.reshape(H, W, C)), interpolation='nearest')
#     print(i)
#     plt.show()
#
# for i, (red_data, orig_data) in enumerate(zip(recon, batch_out)):
#     fig, ax = plt.subplots(1,2)
#     ax[0].imshow(tools.unstandardize_batch(red_data.reshape(H, W, C)), interpolation='nearest')
#     ax[1].imshow(tools.unstandardize_batch(orig_data.reshape(H, W, C)), interpolation='nearest')
#     print(i)
#     plt.show()
#     if i > 10:
#         break
