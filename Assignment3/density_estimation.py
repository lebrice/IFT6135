"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


import numpy as np
import torch 
import matplotlib.pyplot as plt
from q1 import get_optimal_discriminator

import sys
if sys.version_info[:3] < (3, 6, 7):
    print("Please use python 3.6.7 when grading this assignment.")
    print("All The necessary pip packages to install will be listed in 'requirements-pip.txt'")
    if sys.version_info.major == 2:
        print("Python 2? Really?! This is 2019. Come on.")
    exit()

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
plt.savefig("./images/q1_4_1.png")
############### import the sampler ``samplers.distribution4'' 

############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 

to_tensor = lambda x: torch.as_tensor(x).float

from samplers import distribution4, distribution3
f_0 = distribution3(batch_size=512) # standard gaussian
f_1 = distribution4(batch_size=512) # modified 'unknown' distribution

print("Training discriminator...")
discriminator = get_optimal_discriminator(f_1, f_0, maxsteps=1_000, threshold=1e-3)


def estimate_density(xx):
    d_x = discriminator(xx)
    # prevent division by zero:
    d_x = np.maximum(d_x, 1e-8)
    d_x = np.minimum(d_x, 1 - 1e-8)

    base_density = N(xx)
    scaling_factor = d_x / (1 - d_x) 
    
    # plt.plot(xx, scaling_factor, color="red")

    estimate = base_density * scaling_factor
    return estimate

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
plt.savefig("./images/q1_4_2.png")




if __name__ == "__main__":
    plt.show()



