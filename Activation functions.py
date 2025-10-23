#Implement the different activation functions for a sample data.

import numpy as np
import matplotlib.pyplot as plt

relu = lambda z: np.maximum(0,z)
sig = lambda z: 1/(1+np.exp(-z))
tan = np.tanh
lrelu = lambda z,a=0.01: np.where(z>0,z,a*z)
soft = lambda z: np.log1p(np.exp(z))
swish = lambda z: z*act_sig(z)

x_vals = np.linspace(-8,8,400)
activations = {
    "ReLU": relu, "Sigmoid": sig, "Tanh": tan,
    "LeakyReLU": lrelu, "Softplus": soft, "Swish": swish
}

plt.style.use('dark_background')
plt.figure(figsize=(10,6))

colors = plt.cm.viridis(np.linspace(0,1,len(activations)))
for (name,f), c in zip(activations.items(), colors):
    y = f(x_vals)
    plt.plot(x_vals, y, label=name, color=c, lw=2)
    plt.fill_between(x_vals, 0, y, color=c, alpha=0.1, linewidth=0)

plt.axhline(0,color='white',lw=0.5,ls='--')
plt.axvline(0,color='white',lw=0.5,ls='--')
plt.title("Activation Functions (Enhanced)", fontsize=16)
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

X_sample = np.linspace(-3,3,7)
print("Sample X:", X_sample)
for n,f in activations.items(): print(f"{n}: {np.round(f(X_sample),4)}")
