浙江大学脑机智能导论（2024秋冬）实验一参考代码。

包括 tuning curve、$R^2$分布、使用三种方法解码运动的解法。

仓库中的代码均为个人解答，仅作参考。

An example solution for Introduction to Brain-Computer Interface, project 1.
<center>
  <img src="https://github.com/Undermyth/bci-p1/blob/main/pics/r2.png"> 
</center>

[https://github.com/Undermyth/bci-p1/blob/main/movie_slow.mp4](https://github.com/user-attachments/assets/612464a3-62a0-43aa-959f-76a216a945d4)

The solution includes plotting the tuning curve, and using three methods to decode the neuronal activity to predict motion data. Linear regression, Kalman filter and linear filter is used.

Data can be downloaded at [this website](https://zenodo.org/records/3854034).

- Plot the tuning curve: `main.ipynb`
- Neuronal data decoding: `filter.ipynb`
