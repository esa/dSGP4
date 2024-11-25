import torch
import torch.nn as nn

from .util import initialize_tle, propagate, propagate_batch
from torch.nn.parameter import Parameter

class mldsgp4(nn.Module):
    def __init__(self, 
                normalization_R=6958.137, 
                normalization_V=7.947155867983262, 
                hidden_size=100, 
                input_correction=1e-2, 
                output_correction=0.8):
        """
        This class implements the ML-dSGP4 model, where dSGP4 inputs and outputs are corrected via neural networks, 
        better match simulated or observed higher-precision data.

        Parameters:
        ----------------
        normalization_R (``float``): normalization constant for x,y,z coordinates.
        normalization_V (``float``): normalization constant for vx,vy,vz coordinates.
        hidden_size (``int``): number of neurons in the hidden layers.
        input_correction (``float``): correction factor for the input layer.
        output_correction (``float``): correction factor for the output layer.
        """
        super().__init__()
        self.fc1=nn.Linear(6, hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size, 6)
        self.fc4=nn.Linear(6,hidden_size)
        self.fc5=nn.Linear(hidden_size, hidden_size)
        self.fc6=nn.Linear(hidden_size, 6)
        
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.normalization_R=normalization_R
        self.normalization_V=normalization_V
        self.input_correction = Parameter(input_correction*torch.ones((6,)))
        self.output_correction = Parameter(output_correction*torch.ones((6,)))

    def forward(self, tles, tsinces):
        """
        This method computes the forward pass of the ML-dSGP4 model.
        It can take either a single or a list of `dsgp4.tle.TLE` objects, 
        and a torch.tensor of times since the TLE epoch in minutes.
        It then returns the propagated state in the TEME coordinate system. The output
        is normalized, to unnormalize and obtain km and km/s, you can use self.normalization_R constant for the position
        and self.normalization_V constant for the velocity.

        Parameters:
        ----------------
        tles (``dsgp4.tle.TLE`` or ``list``): a TLE object or a list of TLE objects.
        tsinces (``torch.tensor``): a torch.tensor of times since the TLE epoch in minutes.

        Returns:
        ----------------
        (``torch.tensor``): a tensor of len(tsince)x6 representing the corrected satellite position and velocity in normalized units (to unnormalize to km and km/s, use `self.normalization_R` for position, and `self.normalization_V` for velocity).
        """
        is_batch=hasattr(tles, '__len__')
        if is_batch:
            #this is the batch case, so we proceed and initialize the batch:
            _,tles=initialize_tle(tles,with_grad=True)
            x0 = torch.stack((tles._ecco, tles._argpo, tles._inclo, tles._mo, tles._no_kozai, tles._nodeo), dim=1)
        else:
            #this handles the case in which a singlee TLE is passed
            initialize_tle(tles,with_grad=True)
            x0 = torch.stack((tles._ecco, tles._argpo, tles._inclo, tles._mo, tles._no_kozai, tles._nodeo), dim=0).reshape(-1,6)
        x=self.leaky_relu(self.fc1(x0))
        x=self.leaky_relu(self.fc2(x))
        x=x0*(1+self.input_correction*self.tanh(self.fc3(x)))
        #now we need to substitute them back into the tles:
        tles._ecco=x[:,0]
        tles._argpo=x[:,1]
        tles._inclo=x[:,2]
        tles._mo=x[:,3]
        tles._no_kozai=x[:,4]
        tles._nodeo=x[:,5]
        if is_batch:    
            #we propagate the batch:
            states_teme=propagate_batch(tles,tsinces)
        else:
            states_teme=propagate(tles,tsinces)
        states_teme=states_teme.reshape(-1,6)
        #we now extract the output parameters to correct:
        x_out=torch.cat((states_teme[:,:3]/self.normalization_R, states_teme[:,3:]/self.normalization_V),dim=1)

        x=self.leaky_relu(self.fc4(x_out))
        x=self.leaky_relu(self.fc5(x))
        x=x_out*(1+self.output_correction*self.tanh(self.fc6(x)))
        return x

    def load_model(self, path, device='cpu'):
        """
        This method loads a model from a file.

        Parameters:
        ----------------
        path (``str``): path to the file where the model is stored.
        device (``str``): device where the model will be loaded. Default is 'cpu'.
        """
        self.load_state_dict(torch.load(path,map_location=torch.device(device)))
        self.eval()