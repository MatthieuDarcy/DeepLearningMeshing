# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:12:31 2021

@author: matth
"""


import numpy as np


import torch
print(torch.__version__)

from unet_model import UNet



import torch
print(torch.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"


import meshio
from preprocess import preprocess

torch.no_grad
#%%

"""
Modify this

"""
# Modify this to select your model folder
model_path = "/home/boussugef/0_WORK/DATA/MATTHIEU/prod/model_folder_test/"

image_name = "image_test.pt"
im_path = "/home/boussugef/0_WORK/DATA/MATTHIEU/prod/image_folder_test/"




# data location
location ='/home/boussugef/0_WORK/DATA/MATTHIEU/prod/data_folder_test/'
save_location = '/home/boussugef/0_WORK/DATA/MATTHIEU/prod/predict_folder_test/'

# Data size
data_size = 4

#%%

# Load images
images = torch.load(im_path + image_name)
input_dim = images.shape[-1]

#%%

# Load the model

model = UNet(2,2)
model.load_state_dict(torch.load(model_path + "model"))
model.eval()
model = model.to(device)
#%%


if __name__ == "__main__":

    mesh_points, Y_torch, tangents_torch, centroids_torch, grid_ext =  preprocess(location, data_size, input_dim)
    
    
    N = centroids_torch.shape[0]
    #%% Compute closest points
    closest_points = []
    for m in mesh_points:
        # compute the pairwise distance
        
         dist = torch.cdist(m, grid_ext[0])
         
         closest_points.append(torch.min(dist, axis = -1)[-1])
    
    print("Finished data pre-processing")

    #%%
    # Send all quantities to CPU
    centroids_torch = centroids_torch.cpu()
    tangents_torch = tangents_torch.cpu()
    grid_ext = grid_ext.cpu()
    model = model.cpu()


    
     #%%
    
    print("Saving predictions")

    for i in range(1, data_size+ 1):


        filename = "mesh " + str(i) + ".vtu" #"data_25 (" + str(i) + ").vtu" #"Disk" + str(i) + ".vtu"
        #print(filename)
        mesh = meshio.read(location + filename)
            
        # Extract the points
        points = mesh.points
        # Extract the cells
        c = mesh.cells_dict['triangle']

    
        # Forward pass through the network
        
        pred_vf = model.forward(images[i-1:i])
        pred_vf = pred_vf.detach()
        
        print(pred_vf.shape)
        
        closest_batch =  [closest_points[i-1]]
        pred_vf = model.convert_im_to_vf(pred_vf)
        vf_pred = model.interpolate_vf_nearest(closest_batch, pred_vf)
        
        values = mesh.point_data.values()
        v_iterator = iter(values)
        Y = next(v_iterator)
        print(Y.shape, vf_pred[0].numpy().shape)
        
        pred = vf_pred[0].numpy()
        
        #temp = np.zeros(shape = (pred.shape[0], 1))
        #pred = np.vstack(pred, temp)
        
        mesh = meshio.Mesh(
            mesh.points,
            mesh.cells_dict,            
            point_data={"Predicted": pred, "Ground_truth": Y},
            )

        mesh.write(save_location + "mesh_" + str(i)+ ".vtk")
        if i % 100 == 0:
            print(i)
    

    
