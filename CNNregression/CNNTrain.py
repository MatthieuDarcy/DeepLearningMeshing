# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:46:43 2021

@author: matth
"""
import numpy as np


# import matplotlib.pyplot as plt

# plt.plot(np.arange(10))
# plt.close()

import math

import torch
print(torch.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'


from preprocess import preprocess

from helper_functions import gen_batches
    #%%


from PixelRegressionModel import RegressionNet




#%%

epochs =  2
batch_size = 8
train_size = 2#int(input("Size of training set?"))


# Location of the meshes
location ="D:/Documents/University/MVA/CEA/Data/data_folder_test/"
data_size = 4

# Location of the images
im_path = 'D:/Documents/University/MVA/CEA/Data/' + "Images_data/"

# Name of the images
image_name = "image_test.pt"

# Model path
model_path = "D:/Documents/University/MVA/CEA/Data/Code_clean/UNet/models"

# Seed for the train test split 
shuffle_seed = 0
#%%







images = torch.load(im_path + image_name) 

print("Image loaded with shape: ", images.shape)

input_dim = images.shape[-1]
xx = np.linspace(-1.0, 1.0, input_dim)
yy = np.linspace(-1.0, 1.0, input_dim)



model = RegressionNet(2, in_size= input_dim)
#%%


if __name__ == "__main__":

    mesh_points, Y_torch, tangents_torch, centroids_torch, grid_ext =  preprocess(location, data_size, input_dim)
    
    
    N = centroids_torch.shape[0]

        
    #%% Compute the closest points for the nearest interpolator
    
    closest_points = []
    for m in mesh_points:
        # compute the pairwise distance
        
         dist = torch.cdist(m, grid_ext[0])
         
         closest_points.append(torch.min(dist, axis = -1)[-1])
    
    #print("Finished computing nearest interpolator")

    #%%
    # Send all quantities to CPU
    centroids_torch = centroids_torch.to("cpu")
    tangents_torch = tangents_torch.to("cpu")
    grid_ext = grid_ext.to("cpu")
    model = model.to("cpu")


    #%%
    


    
    #%%
    best_loss = math.inf
    def test(model, images_test, Y_test, closest_test, gen_batches, batch_size = 8):
        
        model.eval()
        
        test_loss = 0
        test_size = images_test.shape[0]
        
        batch_idx = gen_batches(test_size, batch_size)

        for b in batch_idx:
            
            batch_size = len(b)
            
            images_batch = images_test[b].to(device)
            Y_batch = [Y_test[i].to(device) for i in b]

            
            vf_pred = model.forward(images_batch)
            vf_pred = model.convert_im_to_vf(vf_pred)
                            
            closest_batch = [closest_test[i] for i in b]
                

            pred = model.interpolate_vf_nearest(closest_batch, vf_pred)
                
            loss = model.loss_function(pred, Y_batch)/images_test.shape[0]
            
            test_loss += loss.item()
        
        del Y_batch, loss, pred
        return test_loss

            
            
    #%%
    
    
    def train(model,train_list, test_list, optimizer, gen_batches, batch_size = 8, epochs = 1):
        """

        Parameters
        ----------
        model : Pytorch model.
        
        
        train_list, test_list: lists containing:
            

        normals : tensor of normals 
        pairwise_diff : pairwise differrence between grid and centroids
        Y : target (vf at mesh points)
        grid_dist : pairwise distance between points on the grid 
        mesh_dist : pairwise distance between points on the grid and the mesh
        
        grid_coord : coordinates of the grid, list of form [xx, yy]
        optimizer : pytorch optimizer 
        epochs : Int. The default is 1.

        Returns
        -------
        None.

        """
        images, Y, mesh, closest_train = train_list
        images_test, Y_test, mesh_test, closest_test = test_list
        train_loss = []
        test_loss = []
        best_loss = math.inf
 
        for i in range(epochs):
            
            # Build a batch schedule
            
            batch_idx = gen_batches(images.shape[0], batch_size)
            temp_loss = 0
            
            
            model.train()
            #print(batch_idx)
            for b in batch_idx:

                #print("b", b)
                batch_size = len(b)
                optimizer.zero_grad()
                
                
                images_batch = images[b].to(device)
                Y_batch = [Y[i].to(device) for i in b]
                
                vf_pred = model.forward(images_batch)
                vf_pred = model.convert_im_to_vf(vf_pred)


                closest_batch = [closest_train[i] for i in b]
                

                pred = model.interpolate_vf_nearest(closest_batch, vf_pred)
 
                #print("prediction", pred)
                loss = model.loss_function(pred, Y_batch)/images.shape[0]
                #print(pred[0].get_device())
                temp_loss += loss.item()
                
                loss.backward()
                
                optimizer.step()

            
            
            del Y_batch, loss
            
            train_loss.append(temp_loss)
            print("Train loss at epoch {}: {}".format(i, temp_loss))
            del temp_loss
            
            t_loss= test(model, images_test, Y_test, closest_test, gen_batches, batch_size = batch_size)
            test_loss.append(t_loss)
            
            print("Test loss at epoch {}: {}".format(i, t_loss))
            if t_loss < best_loss:
                best_loss = t_loss
                torch.save(model.state_dict(), model_path + "model")
                print("Saving model")
        
        print("Best test loss: ", best_loss)
        
        torch.save(model.state_dict(), model_path + "model_final_it")
        return train_loss, test_loss

    #%%
    images = images.float()
    
    
    #%% Shuffle the data
    
    idx_shuffle = torch.randperm(images.shape[0], generator = torch.manual_seed(shuffle_seed))
    
    images_shuffle = images[idx_shuffle]
    Y_shuffle = [Y_torch[i] for i in idx_shuffle]
    mesh_shuffle = [mesh_points[i] for i in idx_shuffle]
    closest_points_shuffle = [closest_points[i] for i in idx_shuffle]
    


    
    #%%
    train_list = [images_shuffle[:train_size], Y_shuffle[:train_size], mesh_shuffle[:train_size], closest_points_shuffle[:train_size]]
    test_list = [images_shuffle[train_size:], Y_shuffle[train_size:], mesh_shuffle[train_size:], closest_points_shuffle[train_size:]]   
    

    
    #%%
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-7)
    model.to(device)

    #%%
    print("Training")
    batch_size = batch_size

    train_loss, test_loss = train(model, train_list, test_list, optimizer, gen_batches, batch_size = batch_size, epochs = epochs)
    
    #%%
    

    print("Saving")

    
    np.save(model_path +"train_loss", np.array(train_loss))
    np.save(model_path +"test_loss", np.array(test_loss))


                
        
    
