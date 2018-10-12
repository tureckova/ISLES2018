## YellowFin for keras!

I made some modifications to the TensorFlow implementation of the [YellowFin](https://github.com/JianGoForIt/YellowFin) optimizer introduced by Jian Zhang, Ioannis Mitliagkas, and Christopher Ré in the paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471).

**My notes on the YellowFin paper and this script are [here](http://nnormandin.com/science/2017/07/01/yellowfin.html) on my website. Additional test examples coming soon**

The YellowFin version here can now be used in keras through the `TFOptimizer()` function. For example:

```python
from yellowfin import YFOptimizer
from keras.optimizers import TFOptimizer

# define your optimizer
opt = TFOptimizer(YFOptimizer())

# compile a classification model
model.compile(loss = 'categorical_crossentropy',
               metrics = ['acc'],
               optimizer = opt)
```

The changes I made to the TensorFlow version of YellowFin allow the `TFOptimizer()` class from keras to directly call the `compute_gradients()` and `apply_gradients()` methods necessary to compile the model object. This could be significantly improved by making a native keras optimizer that inherits directly from the `keras.optimizers.Optimizer` class rather than hacking together some TensorFlow compatibility.

There's currently an example using the CIFAR10 CNN from the keras examples. I've tested this optimizer on Ubuntu 16.04 with both CPU and GPU. 
