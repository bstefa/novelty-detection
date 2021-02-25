'''
Script used to run novelty detection experiment using Principal
Component Analysis.

Uses:
    Module: PCABaseModule
    Model: IncrementalPCA
    Dataset: LunarAnalogueDataGenerator
'''

import numpy as np
import matplotlib.pyplot as plt

from utils import tools, losses
from datasets.lunar_analogue import LunarAnalogueDataGenerator
from models.incremental_pca import IncrementalPCA
from modules.pca_base_module import PCABaseModule


DEFAULT_CONFIG_FILE = 'configs/pca.yaml'
config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

# Initialize datagenerator
datagenerator = LunarAnalogueDataGenerator(config)

# Initialize model. With n_component=None the number of features will
# autoscale to the batch size
model = IncrementalPCA(n_components=None)

# Initialize experimental module
module = PCABaseModule(datagenerator, model, config)

# # Throughput sanity check--ensure everything's running okay
# print('[status] Running pipeline sanity check.')
# module.fit_pca(fast_dev_run=1).transform_pipeline(fast_dev_run=1)

# Incrementally fit training set with PCA
module.fit_pca()

# Run transform pipeline on test set (implicitly called in .transform_generator())
novelty_scores = module.transform_pipeline()

# Finally, an array of novelty scores
print('Number of images processed:', np.array(novelty_scores).shape)

plt.hist(novelty_scores, bins=len(novelty_scores))
plt.show()

#
# #%%
# batch_in, batch_redu = next(module.transform_pca())
# reconstr = module.inverse_transform(batch_redu)
# print(batch_in.shape, reconstr.shape, batch_redu.shape)
#
# #%%
#
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(tools.unstandardize_batch(reconstr[0].reshape(512, 512, 3)), interpolation='nearest')
# ax[1].imshow(tools.unstandardize_batch(batch_in[0]), interpolation='nearest')
# plt.show()
# # After transforming the data, new properties are available, as shown below.
# # See the code for more details.
#
#
# #%%
# print(module.components.shape)
# print(np.cumsum(module.explained_variance))
# # plt.plot(module.explained_variance.cumsum());
# plt.plot((module.explained_variance/sum(module.explained_variance)).cumsum())
# plt.show()
# print(module.explained_variance)
# mean, var = module.meanvar
# print(mean.shape)
# print(var.shape)
# # Think of the components like eigenimages..
# # T
#
#
# #%%
# plt.imshow(mean.reshape((512, 512, 3))); plt.show()
#
#
#
# comp1 = module.components[2]
# plt.imshow(tools.unstandardize_batch(comp1.reshape(512, 512, 3))); plt.show()
#
# print(comp1.mean(), comp1.std())
# gauss_comp = gaussian_window(comp1.mean(), comp1.std())
# gauss = gaussian_window(0, 1)
#
# print(gauss.shape)
# # plt.plot(np.linspace(0 - 3.5*1, 0 + 3.5*1, 100), gauss)
# plt.plot(np.linspace(comp1.mean() - 3.5*comp1.std(), comp1.mean() + 3.5*comp1.std(), 100), gauss_comp)
#
# plt.show()



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
