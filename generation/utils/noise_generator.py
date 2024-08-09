import torch
import numpy as np

class NoiseGenerator:

    def __init__(self):
        super().__init__()

    def getNoise(self, data, variance=0.1, interval=[0,1]): 
        #return noise of size (batch_size, nb_features (config_file), len (as the audio embedding))
        # we maintain temporal consistency in the generated noise 
        begin_noise = torch.randn_like(data[:,:,0].unsqueeze(2)) * variance
        end_noise = torch.randn_like(data[:,:,-1].unsqueeze(2)) * variance
        final_noise = torch.clone(begin_noise)
        current = torch.clone(begin_noise)
        step = (end_noise - begin_noise)/(data.shape[2]-1)
        for _ in range(data.shape[2]-2):
            current = current + step
            final_noise = torch.cat((final_noise, current), dim=2)
        final_noise = torch.cat((final_noise, end_noise), dim=2)

        noisy_data = data + final_noise
        noisy_data = torch.clamp(noisy_data, interval[0], interval[1])
        return noisy_data.to(data)





