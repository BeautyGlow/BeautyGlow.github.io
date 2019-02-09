# BeautyGlow: On-Demand Makeup Transfer Framework with Reversible Generative Network
Implementation of "BeautyGlow: On-Demand Makeup Transfer Framework with Reversible Generative Network". The implementation is based on the [chainer implementation of Glow code](https://github.com/musyoku/chainer-glow).

<img src='../images/intro.png'>

<img src='../images/model.png'>

## Getting Started
### Run
You have to train Glow on your dataset and get the latent vectors encoded from the trained glow model.
Then run the main.py to train the latent vector decomposition matrix

```
python3 main.py non-makeup makeup outfile
```


