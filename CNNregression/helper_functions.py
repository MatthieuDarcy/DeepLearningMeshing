import numpy as np
import math
import torch
from torch.nn import functional as F

import time

def gen_edges(points, cells):
    edges = []
    for i in range(0, cells.size, 4):
        # extract the triangle
        triangle = cells[i+1: i + 4]
        current_points = points[triangle]
        for j in range(3):
            edges.append([current_points[j% 3], current_points[(j+1) % 3]])
            
    return edges

def gen_edges_meshio(points, cells):
    
    edges = []
    
    for c in cells:
        current_points = points[c]
        for j in range(3):
            edges.append([current_points[j% 3], current_points[(j+1) % 3]])
    return edges


def gen_tangents_centroids(edges):
    tangents = []
    centroids = []
    for ed in edges:
        tangents.append(ed[1] - ed[0])
        centroids.append((ed[0]+ed[1])/2)
    return np.array(tangents), np.array(centroids)


def convert_np_to_torch(centroids, normals, grid,  points, X):
    return torch.from_numpy(centroids).float(), torch.from_numpy(normals).float(), torch.from_numpy(grid).float(), torch.from_numpy(points).float(), torch.from_numpy(X).float()


def convert_to_torch(array):
    return torch.from_numpy(array).float()


def pad(tensor, pad_quant):
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
    
    #%%
    
def gen_batches(train_size, batch_size = 64):
    """

    Parameters
    ----------
    train_size : Int
    batch_size : Int. The default is 64.

    Returns
    -------
    batch_idx : List with of batch indices for each batch (as tensors). 

    """
        
    idx = torch.randperm(train_size)

        
    batch_idx = []

    for i in range(math.ceil(train_size/batch_size)):
        batch_idx.append(idx[i*batch_size : (i+1)*batch_size])
            
    return batch_idx

def find_bound(data):
        x_max, x_min = -math.inf, math.inf
        y_max, y_min = -math.inf, math.inf
        for element in data:
        
            x_t_max, y_t_max = np.max(element[0], axis = 0)[0], np.max(element[0], axis = 0)[0]
            x_t_min, y_t_min = np.min(element[0], axis = 0)[0], np.min(element[0], axis = 0)[0]
    
            if x_max < x_t_max:
                x_max = x_t_max
            
            if x_min > x_t_min:
                x_min = x_t_min
                
            if y_max < y_t_max:
                y_max = y_t_max
                
            if y_min > y_t_min:
                y_min = y_t_min
        return x_max, x_min, y_max, y_min

def normalize_data(data, center = 0, bound = 0.8):
        
        for point in data:
            element = point[0]
            # Compute the max and the min
            x_max, y_max = np.max(element, axis = 0)
            x_min, y_min = np.min(element, axis = 0)
            
            #print(x_max, y_max, x_min, y_min)

            # Normalize to [0, 1]
            element[:, 0] = (element[:, 0] - x_min)/(x_max - x_min)
            element[:, 1] = (element[:, 1] - y_min)/(y_max - y_min)
            
            # Translate to the required interval
            
            element[:, 0] -= 0.5
            element[:, 1] -= 0.5
            
            element[:, 0] *= bound/0.5
            element[:, 1] *= bound/0.5
        return data
    
  

    
# Compute the edges and centroids
    
def compute_edges_centroids(data, cells):
        
        #
        edge_list = []
        for i, element in enumerate(data):
            points = element[0]
            c = cells[i]
            
            edges = gen_edges_meshio(points, c)
            edge_list.append(edges)
            tangents, centroids = gen_tangents_centroids(edges)
            
            element.append(tangents)
            element.append(centroids)
            
            if i % 100 == 0:
                print(i)
        return data, edge_list
    
    
def pad_centroids_tangents(data):
        tangents = []
        centroids = []

        num_max = 0
        for e in data:
            if num_max < e[1].shape[0]:
                num_max = e[1].shape[0]
        
        for e in data:
 
            tangents.append(pad(e[1], num_max - e[1].shape[0] ))
            centroids.append(pad(e[2],num_max-  e[2].shape[0] ))
        return tangents, centroids

def convert_data_to_tensor(data, Y):
        new_data = []
        new_Y = []
        for e1, e2 in zip (data, Y):
            
            points, tangents, centroids = convert_to_torch(e1[0]), convert_to_torch(e1[1]), convert_to_torch(e1[2])
            new_data.append([points, tangents, centroids])
            target = convert_to_torch(e2[:, :-1])
            new_Y.append(target)
            
        return new_data, new_Y

#%%
    # Compute the pairwise norm between data points
def norm_matrix( matrix_1, matrix_2):
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

def set_invariants(centroids, normals, grid, batch_size = 8):
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

        for i in range(batches):
            p_diff.append(norm_matrix(centroids[i*batch_size: (i+1)*batch_size], grid[i*batch_size: (i+1)*batch_size]))
            
        pairwise_diff = torch.cat(p_diff)
        return pairwise_diff 

    # Compute the kernel matric 
def kernel_RBF(norm_matrix, parameters):
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
def compute_current_vector_field( normals, pairwise_diff, mu):
        """
        Computes the current vector field.
        
        Input: 
            Centroids: centroids of the mesh elements (ex: centroid of a triangle). Size =  [ N x num of elements]
            Normal: the normal to each of the mesh elements (or the tangents in 2D). Size =  [ N x num of elements]
            Grid: the discretization grid of the ambient space,, size [K x L]
            Mu: the parameter of the RBF kernel (size 1)
        
        Returns:
            tensor of size [N x C x (K x L)]. (K = W, L = H)
            
        """

            
        K = kernel_RBF(pairwise_diff, mu)

        # Repeat values along the axis (we use the expand function because it does nto copy the array)
        K = K[..., None].expand(K.shape[0], K.shape[1], K.shape[2], normals.shape[-1])

        #return K 
        normal = normals[:, :, None, :]
        
        #return torch.multiply(K, normal)
        
        return torch.sum(torch.multiply(K, normal), axis = 1)
    
    
def convert_vf_to_im(vf,grid_coord):
        
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

        vf = (vf - min_vf)/(max_vf - min_vf)
        
        
        # Reshape
        vf = torch.reshape(vf, (vf.shape[0], yy.shape[0], xx.shape[0], vf.shape[-1]))
        
        vf = torch.swapaxes(vf, 1, -1)
        vf = torch.swapaxes(vf, 2, -1)

        return  vf #♣torch.flip(vf, (1,))
    

def compute_current_vector_field_ind( normals, K, batch_size = False, device = "cpu"):
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
        
def compute_im(centroids_torch, tangents_torch, grid_ext, device, grain, batch_size, xx, yy,ind = True, im_batch = False):
    print("Device", device)
    if ind == True:
        
        start = time.time()
        N = centroids_torch.shape[0]
        print("Computing images")
    
        images = []
        #mu = torch.tensor([grain/2]).to(device)
        mu = grain*2
        for i in range(N):
            if i %10 == 0:
                print(i)
            centroids_batch = centroids_torch[i:(i+1)].to(device)
            tangents_batch = tangents_torch[i: (i+1)].to(device)
            grid_batch = grid_ext[i:(i+1)].to(device)
        
        
        

            K = set_invariants(centroids_batch, tangents_batch, grid_batch)     
            del centroids_batch, grid_batch

            
            K /= mu**2
            K = K.to("cpu")
            K = torch.exp(-(K))

            
            vf = compute_current_vector_field_ind(tangents_batch, K, batch_size =im_batch, device = device)
            #print(vf.shape)
            vf = convert_vf_to_im(vf[None, ...],[xx, yy])
            #print(torch.cuda.memory_allocated())
            images.append(vf.to("cpu"))
        
            del(K)
            #print(vf.shape)
            del(vf, tangents_batch) 

      
        end = time.time()
        images = torch.cat(images)
    
        print("Image stored on device", images.get_device())
        print("Image computation:", end - start)
        
    
    elif ind == False:
        
        start = time.time()
        N = centroids_torch.shape[0]
        print("Computing images")
    
        images = []
        #mu = torch.tensor([grain/2]).to(device)
        mu = grain*2
        for i in range(math.ceil(N/batch_size)):
            if i %10 == 0:
                print(i)
            centroids_batch = centroids_torch[i*batch_size:(i+1)*batch_size].to(device)
            tangents_batch = tangents_torch[i*batch_size: (i+1)*batch_size].to(device)
            grid_batch = grid_ext[i*batch_size:(i+1)*batch_size].to(device)
        
        
        

            K = set_invariants(centroids_batch, tangents_batch, grid_batch)     
            del centroids_batch, grid_batch

            
            K = torch.exp(-(K/mu**2))

            vf = compute_current_vector_field(tangents_batch, K, device = device)
            #print(vf.shape)
            vf = convert_vf_to_im(vf,[xx, yy])
            #print(torch.cuda.memory_allocated())
            images.append(vf.to("cpu"))
        
            del(K)
            #print(vf.shape)
            del(vf, tangents_batch) 

      
        end = time.time()
        images = torch.cat(images)
        print(images.get_device())
        print("Image computation:", end - start)
    

    
    return images

                