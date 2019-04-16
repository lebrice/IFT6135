"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))

############### import the sampler ``samplers.distribution4'' 

############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 
from q1 import get_optimal_discriminator

to_tensor = lambda x: torch.as_tensor(x).float

from samplers import distribution4, distribution3
f_0 = distribution3(batch_size=512)
f_1 = distribution4(batch_size=512)

print("Training discriminator...")
discriminator = get_optimal_discriminator(f_0, f_1)
def estimate_density(xx: np.ndarray) -> np.ndarray:
    d_x = discriminator(xx)
    factor = d_x / (1 - d_x)
    # f_0_x = next(distribution3(batch_size=xx.shape[0]))
    f_0_x = xx
    return (f_0_x * factor)

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

# r = xx # evaluate xx using your discriminator; replace xx with the output
r = discriminator(xx) # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')
# estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator; 
#                                # replace "np.ones_like(xx)*0." with your estimate
estimate = estimate_density(xx) # estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')









