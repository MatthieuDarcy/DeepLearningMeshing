# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:12:31 2021

@author: matth

This file will import and run the model and plot some examples.
"""


import numpy as np

import matplotlib.pyplot as plt

import torch
print(torch.__version__)

from unet_model import UNet

from helper_functions import gen_edges, gen_edges_meshio, gen_tangents_centroids, gen_batches, find_bound, normalize_data, compute_edges_centroids
from helper_functions import pad_centroids_tangents, convert_data_to_tensor, convert_np_to_torch, convert_to_torch, pad
from helper_functions import compute_current_vector_field, convert_vf_to_im, set_invariants
import math
from helper_functions import compute_current_vector_field, convert_vf_to_im, set_invariants, compute_im

import torch
print(torch.__version__)
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"



from preprocess import preprocess

torch.no_grad
#%%

"""
Modify this

"""
# Modify this to select your model folder
model_path = "model_path"

image_name = "image_name"
im_path = "im_path"



# data location
location ='data location' 

# Data size
data_size = 1197

#%%

# Load images
images = torch.load(im_path + image_name)
input_dim = images.shape[-1]
#%%
"""
Plot the train and test loss
"""

train_loss = np.load(model_path + "train_loss.npy")
test_loss = np.load(model_path + "test_loss.npy")

it =0
plt.figure()
plt.plot(train_loss[it:])
plt.title("Train loss")

plt.figure()
plt.plot(test_loss[it:])
plt.title("Test loss")




#%%

# Load the model

model = UNet(2,2)
model.load_state_dict(torch.load(model_path + "model"))
model.eval()
model = model.to(device)

#%%

if __name__ == "__main__":

    mesh_points, tangents_torch, centroids_torch, grid_ext =  preprocess(location, data_size, input_dim)
    
    
    N = centroids_torch.shape[0]
    #%% Compute closest points
    closest_points = []
    for m in mesh_points:
        # compute the pairwise distance
        
         dist = torch.cdist(m, grid_ext[:, :, 0])
         
         closest_points.append(torch.min(dist, axis = -1)[-1])
    
    print("Finished data pre-processing")

    #%% Send all quantities to CPU
    centroids_torch = centroids_torch.cpu()
    tangents_torch = tangents_torch.cpu()
    grid_ext = grid_ext.cpu()
    model = model.cpu()

    #%%
    idx = 175
    vf_im = images[idx] 
    print(vf_im.shape)
    vf_im = torch.swapaxes(vf_im, 0, 1)
    vf_im = torch.swapaxes(vf_im, 1, -1)
    vf_im = F.pad(vf_im, (0,1))
        
    plt.figure()
    plt.title("Input vector field viewed as an image")
    plt.imshow(vf_im)
    
    #%%
    idx = 1000
    vf_im = images_2[idx] 
    print(vf_im.shape)
    vf_im = torch.swapaxes(vf_im, 0, 1)
    vf_im = torch.swapaxes(vf_im, 1, -1)
    vf_im = F.pad(vf_im, (0,1))
        
    plt.figure()
    plt.title("Input vector field viewed as an image")
    
    plt.imshow(vf_im)
    #plt.imshow(torch.flip(vf_im, (0,)))
    #plt.imshow(torch.transpose(vf_im, 0, 1))
    #plt.imshow(torch.rot90(vf_im, 2, [0,1]))
    
    # #%%
    
    # """
    # Forward pass
    # """
    

    # idx = 0
    # model = model.eval()
    
    # pred_vf = model.forward(images[idx:idx+1])
    # pred_vf = pred_vf.detach()
    
    # #%%
    

    # vf_im =  model.layers(images[idx:idx+1])
    # vf_im = vf_im[0]
    
    # #%%
    # vf_im = vf_im.detach()
    
    # print(vf_im.shape)
    # vf_im = torch.swapaxes(vf_im, 0, 1)
    # vf_im = torch.swapaxes(vf_im, 1, -1)
    # vf_im = F.pad(vf_im, (0,1))
        
    # plt.figure()
    # plt.title("Input vector field viewed as an image")
    
    # plt.imshow(vf_im)
    #%%
    
    """
    Visualizing the result
    """
    
    idx = 1000
    pred_vf = model.forward(images[idx:idx+1])
    pred_vf = model.convert_im_to_vf(pred_vf)
    pred_vf = pred_vf.detach()
    plt.figure()
    plt.quiver(mesh_torch[:, 0], mesh_torch[:, 1], pred_vf[0, :,0], pred_vf[ 0, :, 1])
    plt.title("Output (on grid)")
    

    #%%
    
    
    closest_batch =  [closest_points[idx]]
    pred = model.interpolate_vf_nearest(closest_batch, pred_vf)
    #pred = model.regress_vf(pred_vf, mesh_torch, [mesh_points[idx]])
    #%%
    t_mesh = mesh_points[idx]
    
    pred_vf = pred[0]
    
    Y_true = Y_torch[idx]
    plt.figure()
    plt.quiver(t_mesh[:, 0], t_mesh[:, 1], Y_true[:,0], Y_true[:, 1], color = "blue", label = "Ground truth")
    plt.title("Ground truth")
    
    #%%
    plt.figure()
    plt.quiver(t_mesh[:, 0], t_mesh[:, 1], pred_vf[:,0], pred_vf[:, 1])
    plt.title("Output (on mesh)")
    #%%
    
    plt.figure()
    
    plt.quiver(t_mesh[:, 0], t_mesh[:, 1], Y_true[:,0], Y_true[:, 1], color = "red", label = "Ground truth")
    plt.quiver(t_mesh[:, 0], t_mesh[:, 1], pred_vf[:,0], pred_vf[:, 1], label = "Predicted")
    
    plt.title("Output vs ground truth")
    plt.legend(loc='lower right')
    
    
    #%%
    plt.figure()
    
    Y_true_norm = Y_true/np.linalg.norm(Y_true, axis = -1)[:, None]
    pred_norm = pred_vf/np.linalg.norm(pred_vf, axis = -1)[:, None]
    plt.quiver(t_mesh[:, 0], t_mesh[:, 1], Y_true_norm[:,0], Y_true_norm[:, 1], color = "red", label = "Ground truth")
    plt.quiver(t_mesh[:, 0], t_mesh[:, 1], pred_norm[:,0], pred_norm[:, 1], label = "Predicted")
    
    plt.title("Output vs ground truth, normalized")
    plt.legend(loc='lower right')
    # #%%
    # #%%
    
    # a = torch.rand(size = (4, 16384, 2))
    # t = model.convert_vf_to_im_no_std(a, [xx, yy])
    
    # a_recov = model.convert_im_to_vf(t)
    
    
    # print(torch.allclose(a, a_recov))
    # #%%
    
    # #idx = [2:4]
    
    # Y = Y_torch[:4]
    # t_mesh = mesh_points[:4]
    # closest = []
    # for t in t_mesh:
    #     dist = torch.cdist(t, mesh_torch)
    #     closest.append(torch.min(dist, axis = 0)[-1])
    
    # #%%

    # def interpolate_vf_nearest(closest_points, Y_pred):
        
    #     """
        
    #     closest_point: list of the closest points of the grid mesh to the tetmesh
        
    #     Y_pred: array of the predicted vf by the network on the grid mesh
    #     """
        
    #     Y_regress = []
        
    #     for c, y in zip(closest_points, Y_pred):
    #         Y_regress.append(y[c])
        
    #     return Y_regress
    
    # pred_on_mesh = interpolate_vf_nearest(closest, Y)
    
    # #%%
    # idx = 450
    # pred = pred_on_mesh[idx]
    # t = t_mesh[idx]
    # plt.figure()
    # plt.scatter(t[:, 0], t[:, 1], c= "red", alpha = 0.1)
    # plt.quiver(mesh_torch[:, 0], mesh_torch[:, 1], pred[:,0], pred[ :, 1])
    
    # #%%
    # plt.figure()
    # plt.quiver(t_mesh[:, 0], t_mesh[:, 1], Y[:,0], Y[:, 1])
    # #plt.scatter(mesh_torch[:, 0], mesh_torch[:, 1], c= "red", alpha = 0.1)
    
    # #%%
    
    # a = torch.tensor([3.0,1])[:, None]
    # b = torch.tensor([-5,0.0])[:, None]
    # loss = L1Loss(reduction = "sum")
    # print(loss(a,b))
    # #%%
    # pred_1 = vf_pred[0]
    
    # # Write mesh 
    
    # mesh = meshio.Mesh(
    # mesh.points,
    # mesh.cells_dict,
    # # Optionally provide extra data on points, cells, etc.
    # point_data={"VF": pred_point_data},
    # )
    
    # #%%
    
    # mesh.write("mesh_50.vtk")
    # #%%
    
    # """
    # Forward pass
    # """
    # tangents_b, pairwise_diff_b, Y_b, mesh_dist_b = tangents_torch[idx][None, :, :], pairwise_diff[idx][None, :, :], Y_torch[idx][None, :, :], mesh_dist[idx][None, :, :]
    # model = model.eval()
    # pred_vf = model.forward(tangents_b, pairwise_diff_b, [xx, yy])
    # pred_vf = pred_vf.detach()
    
    # #%%
    # plt.figure()
    # plt.quiver(mesh_torch[:, 0], mesh_torch, pred_vf[0,:,0], pred_vf[0, :, 1])
    
    
    # #%%
    
    # """
    # Predicted vf
    # """
    
    # pred = model.regress_vf(grid_dist_torch.cpu(), mesh_dist_b, pred_vf)
    # pred = pred[0].detach()
    
    # plt.figure()
    # #plt.scatter(data[idx][0][:, 0], data[idx][0][:, 1])
    # plt.quiver(data[idx][0][:, 0], data[idx][0][:, 1], Y_torch[idx][:, 0], Y_torch[idx][:, 1], color = "red", label = "True vector field")

    # plt.quiver(data[idx][0][:, 0], data[idx][0][:, 1], pred[:, 0], pred[:, 1], label = "Predicted vector field")
    # plt.legend()
    # plt.title("Predicted vector field vs true vector field")
    # #%%
    
    # pred_im = model.convert_vf_to_im(pred_vf, [xx, yy])
    
    
    # pred_im = torch.swapaxes(pred_im[0], 0, 1)
    # pred_im = torch.swapaxes(pred_im, 1, -1)
    # pred_im = F.pad(pred_im, (0,1))
    
    
    # plt.figure()
    # plt.title("Output vector field viewed as an image")
    
    
    # plt.imshow(torch.flip(pred_im, (0,)))
    
    
    
    
    
    
    
    
        