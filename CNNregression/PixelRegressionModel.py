# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:12:47 2021

@author: matth
"""


from torch import nn
from torch.nn import functional as F
import torch 

import torch.optim as optim
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Using gpu: %s ' % torch.cuda.is_available())
#%%

class RegressionNet(nn.Module):
    def __init__(self, in_channels, in_size = 32):
        super(RegressionNet, self).__init__()


        self.in_size = in_size
        
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64,
                              kernel_size= 9, stride= 1, padding  = "same"),

                    nn.BatchNorm2d(64),
    
                    nn.LeakyReLU(),
                    nn.Dropout(0.1),
                    
            nn.Conv2d(64, out_channels=32,
                              kernel_size= 5, stride= 1, padding  = "same"),
            
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.Dropout(0.1),
            
            nn.Conv2d(32, out_channels=32,
                              kernel_size= 5, stride= 1, padding  = 2),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.Dropout(0.1),


            nn.Conv2d(32, out_channels=in_channels,
                              kernel_size= 5, stride= 1, padding  = "same"),
                    nn.BatchNorm2d(in_channels)
            
            
            
            )
    
    
    # Compute the pairwise norm between data points
    def norm_matrix(self, matrix_1, matrix_2):
        """
        Compute the pairwise norm between two matrices

        Parameters
        ----------
        matrix_1 : tensor of size [N x P x 2] (centroids in our case).(P is the size of the largest mesh).
        matrix_2 : Tensor of size [N x (K x L) x 2] (Grid). 

        Returns
        -------
        norm_diff : Tensor of size [N x P x (K x L)].

        """

        norm_diff = torch.cdist(matrix_1, matrix_2)**2
        return norm_diff
    
    def set_invariants(self, centroids, normals, grid, batch_size = 8):
        """
        
        Sets the invarariant quantity (pariwise difference of centroids/normals) and the 
        normals which do not have to be recomputed each time we upodate the parameter
        sigma of the RBF kernel.
        
        Parameters
        ----------
        centroids :  Tensor of size [N x P x 2]
        normals : Tensor of size [N x P x 2]
        grid : Tensor of size [N x (K x L) x 2]
        Returns
        
        -------
        Pairwise norm of difference between grid and centroids
        Pairwise norm of difference between grid and mesh

        """
        
        batches = int(math.ceil(centroids.shape[0]/ batch_size))
        p_diff = []
        #p_mesh = []

        for i in range(batches):
            p_diff.append(self.norm_matrix(centroids[i*batch_size: (i+1)*batch_size], grid[i*batch_size: (i+1)*batch_size]))
            
        pairwise_diff = torch.cat(p_diff)
        return pairwise_diff
    
    def compute_grid_mesh_dist(self, grid, mesh_points):
        
        return self.norm_matrix(grid, mesh_points)
    
    # Compute the kernel matric 
    def kernel_RBF(self, norm_matrix, parameters):
        """
        Compute the gaussian kernel matrix

        Parameters
        ----------
        matrix_1 : Tensor of size [N x P x 2] (centroids in our case).(P is the size of the largest mesh).
        matrix_2 : Tensor of size [N x (K x L) x 2] (Grid). 
        parameters : Tensor of size 1. Parameter of the RBF kernel

        Returns
        -------
        K : Tensor of size [N x P x (K x L)].

        """
        #matrix = self.norm_matrix(matrix_1, matrix_2)
        K =  torch.exp(-norm_matrix/ (parameters**2))
    
        return K
    


    def compute_current_vector_field( self, normals, K):
        """
        Computes the current vector field.
        
        Input: 
            Centroids: centroids of the mesh elements (ex: centroid of a triangle). Size =  [ N x num of elements]
            Normal: the normal to each of the mesh elements (or the tangents in 2D). Size =  [ N x num of elements]
            Grid: the discretization grid of the ambient space,, size [K x L]
            Mu: the parameter of the RBF kernel (size 1)
            Batch_size : size of batch size to sepearete the computation into 
        Returns:
            tensor of size [N x C x (K x L)]. (K = W, L = H)
            
        """

            
        
        
        # Repeat values along the axis (we use the expand function because it does not copy the array)
        K = K[..., None].expand(K.shape[0], K.shape[1], K.shape[2], normals.shape[-1])

        #return K 
        normal = normals[:, :, None, :]
        

        
        return torch.sum(torch.multiply(K, normal), axis = 1)
    
    def compute_current_vector_field_ind( self, normals, K, batch_size = False):
        """
        Computes the current vector field.
        
        Input: 
            Centroids: centroids of the mesh elements (ex: centroid of a triangle). Size =  [ N x num of elements]
            Normal: the normal to each of the mesh elements (or the tangents in 2D). Size =  [ N x num of elements]
            Grid: the discretization grid of the ambient space,, size [K x L]
            Mu: the parameter of the RBF kernel (size 1)
            Batch_size : size of batch size to sepearete the computation into 
        Returns:
            tensor of size [N x C x (K x L)]. (K = W, L = H)
            
        """


        # Repeat values along the axis (we use the expand function because it does nto copy the array)
        K = K[..., None].expand(K.shape[0], K.shape[1], K.shape[2], normals.shape[-1])
        
        #return K 
        normal = normals[:, :, None, :]
        #print(normal.shape)
        vf = []
        
        if batch_size == False:
            return torch.sum(torch.multiply(K.to(device), normal), axis = 1).to("cpu")
        else:

            for i in range(math.ceil(K.shape[-2]/batch_size)):
                K_temp = K[:, :, i*batch_size: (i+1)*batch_size, :]
                temp = torch.sum(torch.multiply(K_temp.to(device), normal), axis = 1).to("cpu")
                #print(temp.shape)
                vf.append(torch.squeeze(temp, dim = 0))

            return torch.cat(vf)

        
        #return torch.sum(torch.multiply(K, normal), axis = 1)
    
    def pad(self, tensor, pad_quant):
        """
        Pads the given tensors with zeros. Allows to make all normals and centroids 
        the same shape and hence allows for more efficient comuputation in the 
        kernel matrix part, without changing the computed value. 
        
        Parameters
        ----------
        tensor : tensor (in this case the normals or centroids) to be padded.
        pad_quant : scalar. How many zeros to add.

        Returns
        -------
        TYPE
            Tensor with added zeros on the second axis, in the final position.

        """
        return F.pad(tensor, pad = (0,0, 0, pad_quant))
    
    def convert_vf_to_im(self, vf,grid_coord):
        
        """
        Transforms a current vector field to an image (for convolution).
        
        Input: 
            vf: vector field of size [N x C x (K x L)]
            grid_coord: list of the xx and yy coordinates the discretization grid
            
        Returns:
            Tensor of size [N x C x L x K]
        
        """
        xx, yy = grid_coord[0], grid_coord[1]
        
        # Normalize data between 0 and 1
        min_vf = torch.min(vf, axis = 1)[0][:, None, :]
        max_vf = torch.max(vf, axis = 1)[0][:, None, :]
        #print(min_vf.shape, max_vf.shape)
        #print(min_vf, max_vf)
        vf = (vf - min_vf)/(max_vf - min_vf)
        
        
        # Reshape
        vf = torch.reshape(vf, (vf.shape[0], yy.shape[0], xx.shape[0], vf.shape[-1]))
        
        vf = torch.swapaxes(vf, 1, -1)
        vf = torch.swapaxes(vf, 2, -1)

        return  vf #♣torch.flip(vf, (1,))
    
    def convert_vf_to_im_no_std(self, vf,grid_coord):
        
        """
        Transforms a current vector field to an image (for convolution).
        
        Input: 
            vf: vector field of size [N x C x (K x L)]
            grid_coord: list of the xx and yy coordinates the discretization grid
            
        Returns:
            Tensor of size [N x C x L x K]
        
        """
        xx, yy = grid_coord[0], grid_coord[1]

        
        
        # Reshape
        vf = torch.reshape(vf, (vf.shape[0], yy.shape[0], xx.shape[0], vf.shape[-1]))
        
        vf = torch.swapaxes(vf, 1, -1)
        vf = torch.swapaxes(vf, 2, -1)

        return  vf #♣torch.flip(vf, (1,))
    
    
    def convert_im_to_vf(self, vf_im):
        vf = vf_im.reshape(vf_im.shape[0], vf_im.shape[1],  vf_im.shape[2]*vf_im.shape[3])

        vf = torch.swapaxes(vf, 1, -1)

        return vf
    
    def forward(self,vf):
        """
        Computes one forward pass through the network.

        Parameters
        ----------
        grid : Tensor of dimension [N x K x L] (repeated identically across dim N)
        normal : Tensor of dimension [N x P x 2] (normals of the elements. 
                                                P is the size of the largest mesh. All others must be padded with 0)
        centroids : Tensor of dimension [N x P x 2]. (centroids of hte elements. P is as above)
        grid_coord : List of 2 tensors corresponding to x and y of the grid. 

        Returns
        -------
        list
            DESCRIPTION.

        """

        im = self.layers(vf)
        
        
        return im
    
    
    def interpolate_vf_nearest(self, closest_points, Y_pred):
        
        """
        
        closest_point: list of the closest points of the grid mesh to the tetmesh
        
        Y_pred: array of the predicted vf by the network on the grid mesh
        """
        
        Y_regress = []
        
        for c, y in zip(closest_points, Y_pred):
            Y_regress.append(y[c])
        
        return Y_regress
    

            
        
        
    
    def loss_function(self, output, target):
        
        loss = torch.nn.MSELoss()
        #loss = torch.nn.L1Loss(reduction = "mean")
        total_loss = 0
        
        for pred, truth in zip(output, target):

            total_loss += loss(pred, truth)
            
        return total_loss
#%%



