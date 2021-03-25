# Lunar Lander
Solving LunarLander-v2 using DQL


![Lunar Lander Gif](https://github.com/prtpydv/lunar-lander/blob/main/gif/LunarLander-v2.gif)


## Getting Started

To get a local copy up and running follow these steps.

### Dependencies

* Python 3.8
* NumPy
* Matplotlib
* TensorFlow Keras 
* Gym 
* Box2D



### Installation

1. Clone the repo
```sh
git clone git@github.com:prtpydv/lunar-lander.git
```
2. Install the prerequisite packages

```sh
pip install numpy
pip install matplotlib
pip install Keras
pip install gym
pip install Box2D gym
``` 


3. Run `dql.py`


### Performance
Our agent solves the LunarLander-v2 environment in approximately 500 steps.
<img src="https://github.com/prtpydv/lunar-lander/blob/main/img/fig%201.png">

After completing the training, our agent was able to sustain a reward of 200 over a hundred episodes.
<img src="https://github.com/prtpydv/lunar-lander/blob/main/img/fig%202.png">

### References
Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
