# ApproxReLU
Unofficial Tensorflow/Keras implementation of ApproxReLU (https://arxiv.org/abs/1906.01975)

## To use:
In Keras, `model.add(ApproxReLU(k=0.5, n=1.3))` or `x = ApproxReLU(k=0.5, n=1.3)(x)`

## Reference:
`@article{saha2019evolution,
  title={Evolution of Novel Activation Functions in Neural Network Training with Applications to Classification of Exoplanets},
  author={Saha, Snehanshu and Nagaraj, Nithin and Mathur, Archana and Yedida, Rahul},
  journal={arXiv preprint arXiv:1906.01975},
  year={2019}
}`

