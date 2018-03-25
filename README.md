Reparameterized Random Features
===========================

This Python code implements Reparameterized Random Features (RRF), presented in the paper "Large-scale Online Kernel Learning with Random Feature Reparameterization" accepted at the 26th International Joint Conference on Artificial Intelligence (IJCAI), 2017.

The code is tested on Windows-based operating system with Python 3.0. Please make sure that you have installed *python-numpy* and *sklearn* to run the example.

Run the demo using this command
-------------------------------------
	python run_rrf.py

Citation
--------

```
@InProceedings{tu_etal_ijcai17_rrf,
  author    = {Tu Dinh Nguyen and Trung Le and Hung Bui and Dinh Phung},
  title     = {Large-scale Online Kernel Learning with Random Feature Reparameterization},
  booktitle = {Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2017},
  pages     = {2543--2549},
  abstract  = {A typical online kernel learning method faces two fundamental issues: the complexity in dealing with a huge number of observed data points (a.k.a the curse of kernelization) and the difficulty in learning kernel parameters, which often assumed to be fixed. Random Fourier feature is a recent and effective approach to address the former by approximating the shift-invariant kernel function via Bocher’s theorem, and allows the model to be maintained directly in the random feature space with a fixed dimension, hence the model size remains constant w.r.t. data size. We further introduce in this paper the reparameterized random feature (RRF), a random feature framework for large-scale online kernel learning to address both aforementioned challenges. Our initial intuition comes from the so-called ‘reparameterization trick’ [Kingma and Welling, 2014] to lift the source of randomness of Fourier components to another space which can be independently sampled, so that stochastic gradient of the kernel parameters can be analytically derived. We develop a well-founded underlying theory for our method, including a general way to reparameterize the kernel, and a new tighter error bound on the approximation quality. This view further inspires a direct application of stochastic gradient descent for updating our model under an online learning setting. We then conducted extensive experiments on several large-scale datasets where we demonstrate that our work achieves state-of-the-art performance in both learning efficacy and efficiency.},
  code      = {https://github.com/tund/RRF},
  doi       = {10.24963/ijcai.2017/354},
  url       = {https://doi.org/10.24963/ijcai.2017/354},
}
```
