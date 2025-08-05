Variational Autoencoder (VAE)
This project implements a Variational Autoencoder (VAE) using PyTorch, with a convolutional encoder-decoder architecture. The model is trained on grayscale image data and is capable of generating new samples by sampling from the learned latent space.

Features
Convolutional Encoder and Decoder

Reparameterization Trick for backpropagation through sampling

Multivariate Gaussian Prior

Visualization of reconstructions and generated samples

Train/test split and model evaluation

Requirements
Python 3.7+

PyTorch

torchvision

scikit-learn

matplotlib

tqdm

Install requirements via:
pip install torch torchvision scikit-learn matplotlib tqdm

Running the Notebook
Clone the repository or download the notebook.

Launch Jupyter Notebook or JupyterLab.

Open VAE.ipynb and run all cells.

Usage
The notebook trains a convolutional VAE on image data.

After training, it visualizes:

Reconstructions of input images.

New samples drawn from the latent space.
