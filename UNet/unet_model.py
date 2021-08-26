""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from unet_parts import *
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


device = "cpu"
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


    
    def loss_function(self, output, target, name = "mse", reg = 0.1):
        
        if name == "mse":
            loss = torch.nn.MSELoss(reduction = "sum")
            #loss = torch.nn.L1Loss(reduction = "mean")
            total_loss = 0
            
            for pred, truth in zip(output, target):
    
                total_loss += loss(pred, truth)/pred.shape[0]
                
                
        elif name == "cosine":
            total_loss = 0
            l = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            for pred, truth in zip(output, target):
                total_loss += 1-torch.mean(l(pred, truth))
                
        elif name == "hybrid":
            total_loss = 0
            loss = torch.nn.MSELoss(reduction = "mean")
            l = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            for pred, truth in zip(output, target):
                total_loss += 1-torch.mean(l(pred, truth)) + reg*loss(pred, truth)

        return total_loss
    
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
    
    def interpolate_vf_nearest(self, closest_points, Y_pred):
        
        """
        
        closest_point: list of the closest points of the grid mesh to the tetmesh
        
        Y_pred: array of the predicted vf by the network on the grid mesh
        """
        
        Y_regress = []
        
        for c, y in zip(closest_points, Y_pred):
            Y_regress.append(y[c])
        
        return Y_regress
    
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

        vf = (vf - min_vf)/(max_vf - min_vf)
        
        
        # Reshape
        vf = torch.reshape(vf, (vf.shape[0], yy.shape[0], xx.shape[0], vf.shape[-1]))
        
        vf = torch.swapaxes(vf, 1, -1)
        vf = torch.swapaxes(vf, 2, -1)

        return  vf #♣torch.flip(vf, (1,))
    
    
    
    def convert_im_to_vf(self, vf_im):
        vf = vf_im.reshape(vf_im.shape[0], vf_im.shape[1],  vf_im.shape[2]*vf_im.shape[3])

        vf = torch.swapaxes(vf, 1, -1)

        return vf
    
    
#%%
if __name__ == "__main__":
    a = torch.rand(size = (10, 2, 128, 128 ))
    b = torch.rand(size = (10, 2, 128, 128))
    
    model = UNet(2, 2)
    
    result = model.forward(a)
    
    l = model.loss_function(a, b, name ="hybrid")/a.shape[0]
    
    print(l)
    
    
    
    
    