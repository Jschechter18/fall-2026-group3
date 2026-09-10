"""
TODO: ISRAEL -> this store must be implemented in order for the SAE to ever get access to the data. The implementation details are up to you. This is really just meant to be an API

I created this as a placeholder for now. This should be replaced though
"""

import torch

class ActivationStore:
    def __init__(self, location: str):
        self.store_location = location # this is likely going to just be a web address to an S3 bucket. We won't use a database, we just need persistent storage
        
        # you also are going to want to cache these activations somewhere. this should be a good task for you
    
    def load_activations(self) -> torch.Tensor:
        num_activation_vectors = 2 # this will be how many 
        input_dims = 4 # this will be the dimensionality of each activation vector, right now it is hardcoded, but you may actually need to determine this dynamically based on the data
        return torch.rand(num_activation_vectors, input_dims)