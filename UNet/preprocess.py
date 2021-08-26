# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:46:43 2021


This code pre-processes your data: imports all the meshes and normalizes, computes
centroids etc...
@author: matth
"""



import numpy as np



import meshio

from helper_functions import  convert_to_torch, normalize_data, compute_edges_centroids, convert_data_to_tensor, pad_centroids_tangents

import torch



def preprocess(location, data_size, input_dim):
    
    

    data = []
    cells = []
    target = []

    print("Importing data")


    for i in range(1, data_size+ 1):


 
        filename = "mesh " + str(i) + ".vtu"

        #print(filename)
        mesh = meshio.read(location + filename)

        # Extract the points
        points = mesh.points
        points = points[:, :-1]

        # Extract the cells
        c = mesh.cells_dict['triangle']

        values = mesh.point_data.values()
        v_iterator = iter(values)
        Y = next(v_iterator)


        data.append([points])
        cells.append(c)


        target.append(Y)

        if i % 100 == 0:
            print(i)


    #%%
    
    print("Normalizing data")
    data = normalize_data(data,  center = 0, bound = 0.9)



    
    #%%
    print("Computing edges and centroids")
    data, edges = compute_edges_centroids(data, cells)
    
    """
    Data is a list of size N (N points). Each element is a list of size 3
    Each element contains (in order): points, normals/tangents, centroids.
    """

    #%% Convert arrays to tensors
    
    data_torch, Y_torch = convert_data_to_tensor(data, target)
    del data, target
    
    
    #%% Pad 
    
    print("Padding arrays")
    tangents, centroids = pad_centroids_tangents(data_torch)
            
    
    #%% Transform the quantities to torch tensors
    tangents_torch = torch.stack(tangents)
    centroids_torch = torch.stack(centroids)
    
    mesh_points = [data_torch[i][0] for i in range(len(data_torch))]
    

    
    #%% Generate the grid
    

 
    xx = np.linspace(-1.0, 1.0, input_dim)
    yy = np.linspace(-1.0, 1.0, input_dim)
    
   

    grid = np.array(np.meshgrid(xx, yy, sparse=False, indexing='ij'))

    
    grid = grid.T.reshape(-1, 2)
    grid_torch = convert_to_torch(grid)
    grid_ext = grid_torch[None, ...].expand(tangents_torch.shape[0], grid_torch.shape[0], grid_torch.shape[1])
    print("Finished data pre-processing")
    
    return mesh_points, Y_torch, tangents_torch, centroids_torch, grid_ext

                
                
            
if __name__ == "__main__":
    
       # Location of the meshes
    location = "D:/Documents/University/MVA/CEA/Data/data_folder_test/"
    data_size = 3
    input_dim = 32
    mesh_points, Y_torch, tangents_torch, centroids_torch, grid_ext =  preprocess(location, data_size, input_dim)
    

                
                
                
                
        
    
