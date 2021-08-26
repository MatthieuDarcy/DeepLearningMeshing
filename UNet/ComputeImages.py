# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:46:43 2021


This code pre-processes your data: i.e. transforms meshes into images in a 
pytorch (.pt) format 
@author: matth
"""



import numpy as np


#import matplotlib.pyplot as plt

#plt.plot(np.arange(10))
#plt.close()




import torch
print(torch.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from preprocess import preprocess

from helper_functions import compute_im


#%%

"Modify these parameters for your use"
im_name = "image_test.pt"

# Size of the  image
input_dim = 32

# How many data points
data_size = 4

# Location of your data
location = "/home/boussugef/0_WORK/DATA/MATTHIEU/prod/data_folder_test/"

# Where to save
save_path = "/home/boussugef/0_WORK/DATA/MATTHIEU/prod/image_folder_test/"

#%%

# General parameters: ind set to True means every image is computed individually (necessary for large image size)
im_batch = 64
ind = True
batch_size = 1 # IMPORTANT: set to 1 unless ind is False



#%%

xx = np.linspace(-1.0, 1.0, input_dim)
yy = np.linspace(-1.0, 1.0, input_dim)
grain = xx[1] - xx[0]



#%%


if __name__ == "__main__":

    
    
    #%%
    
    
    mesh_points, Y_torch, tangents_torch, centroids_torch, grid_ext=  preprocess(location, data_size, input_dim)
    
    N = centroids_torch.shape[0]
    #%%
    # Send all quantities to CPU
    centroids_torch = centroids_torch.to("cpu")
    tangents_torch = tangents_torch.to("cpu")
    grid_ext = grid_ext.to("cpu")



    #%%


    images = compute_im(centroids_torch, tangents_torch, grid_ext, device,
                        grain, batch_size, xx, yy, ind = ind, im_batch = im_batch)


    print("Are any value infinity?: ", torch.isnan(images).any())
    images = images.float()
    print("Image size", images.shape)
    torch.save(images,save_path + im_name)
                
                
            
            
                
                
                
                
        
    
