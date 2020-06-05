---
title: "单脉冲测向中的最大似然方法"
date: 2020-06-05T20:14:05+08:00
categories:
- 阵列信号处理
tags:
- 单脉冲
keywords:
- 最大似然
#thumbnailImage: //example.com/image.jpg
---

<!-- 内容放在这里 -->
&ensp;&ensp;&ensp;&ensp;最大似然估计是一种常用的估计方法，其核心在于利用已有的观测样本建立起似然函数 \\(\mathcal{L}(\theta)\\)，然后求该似然函数的最大解，得到待估计参数 \\(\hat{\theta}=\text{argmax} \mathcal{L}(\theta)\\)。Nickel 在 1993 年将该方法应用到了单脉冲系统中，该方法的优势在于，不受限于具体的阵列形式，只要能够知道干扰叠加噪声的统计特性，就可以利用该方法进行求解。但同时，由于 Nickel 在原文中采用了牛顿梯度法进行求解，因此对于存在靠近波束指向的主瓣干扰时，估计偏差较大，详情可参见参考文献。

<!--more-->

## 最大似然角度估计的模型建立
&ensp;&ensp;&ensp;&ensp;首先，假设阵元个数为 \\(M\\)，我们考虑阵列接收信号的模型为
$$\boldsymbol{z}=b\boldsymbol{a}(\theta)+\boldsymbol{n} \tag{1}$$
上式中，复数 \\(b\\) 表示信源的复振幅，向量 \\(\boldsymbol{a}(\theta)\in\mathbb{C}^{M \times 1}\\) 表示入射角度为 \\(\theta\\) 的导向向量，向量 \\(\boldsymbol{n}\in\mathbb{C}^{M \times 1}\\) 表示各接收阵元的噪声（或干扰叠加噪声），向量 \\(\boldsymbol{z}\in\mathbb{C}^{M \times 1}\\) 表示个阵元接收到的数据。
&ensp;&ensp;&ensp;&ensp;若假设噪声向量服从复高斯分布 \\(\boldsymbol{n}\sim\mathcal{CN}(0,\boldsymbol{Q})\\) 则可知阵列接收数据的概率分布为 \\(\boldsymbol{z}\sim\mathcal{CN}(b\boldsymbol{a}(\theta),\boldsymbol{Q})\\)，进一步我们可以得到其概率密度函数为
$$p(\boldsymbol{z}|\theta,b)=\pi^{-M}|Q|^{-1}\exp\left[-\[\boldsymbol{z}-b\boldsymbol{a}(\theta)\]^H\boldsymbol{Q}^{-1}\[\boldsymbol{z}-b\boldsymbol{a}(\theta)\]\right] \tag{2}$$
然后我们先对该密度函数取负对数似然，并去掉无关的常数部分，得到
$$S(\theta,b)=-\mathcal{L}(\theta,b)=\[\boldsymbol{z}-b\boldsymbol{a}(\theta)\]^H\boldsymbol{Q}^{-1}\[\boldsymbol{z}-b\boldsymbol{a}(\theta)\] \tag{3}$$
首先对上式中的参数 \\(b\\) 求最小二乘解得到
$$b=\[\boldsymbol{a}^H(\theta)\boldsymbol{Q}^{-1}\boldsymbol{a}(\theta)\]^{-1}\boldsymbol{a}^H(\theta)\boldsymbol{Q}^{-1}\boldsymbol{z}$$
将上式代入 \\(S(\theta,b)\\) 并去掉所有常数项，可得
$$S(\theta)=\boldsymbol{z}^{H} \boldsymbol{Q}^{-1} \boldsymbol{a}(\theta)\left[\boldsymbol{a}^H(\theta) \boldsymbol{Q}^{-1} \boldsymbol{a}(\theta)\right]^{-1} \boldsymbol{a}^H(\theta) \boldsymbol{Q}^{-1} \boldsymbol{z} \tag{4}$$
此时，我们可以定义自适应和波束权向量 \\(\boldsymbol{w}(\theta)\\) 为
$$\boldsymbol{w}(\theta)=\left[\boldsymbol{a}^{H}(\theta) \boldsymbol{Q}^{-1} \boldsymbol{a}(\theta)\right]^{-1 / 2} \boldsymbol{Q}^{-1} \boldsymbol{a}(\theta) \tag{5}$$
由此，可以将 \\(S(\theta)\\) 重写为
$$S_\text{scan}(\theta)=|\boldsymbol{w}^H(\theta)\boldsymbol{z}|^2 \tag{6}$$
上式称为最大似然意义下的自适应角度扫描方向图。

## 似然函数的求解
&ensp;&ensp;&ensp;&ensp;接下来，我们令 \\(F(\theta)=\ln\[S_\text{scan}(\theta)\]\\)。这样，波达方向的估计值 \\(\hat{\theta}\\) 可由下式给出

$$\hat{\theta}=\text{argmax}_\theta F(\theta) \tag{7}$$
接着，本文的求解步骤可能会与原论文有些出入，但原理都是一样的。本文直接利用牛顿梯度法进行求解，即
$$\boldsymbol{\theta}^{(\text{new})}=\boldsymbol{\theta}^{(\text{old})}-\boldsymbol{H}^{-1}\nabla F(\boldsymbol{\theta}) \tag{8}$$
注意到，这里的待估计参数变为了向量 \\(\boldsymbol{\theta}\\)，这是因为，实际阵列的参考系可能是面阵这类含有不止一个角度变量的估计问题。上式中，\\(\boldsymbol{H}=\nabla\nabla F(\boldsymbol{\theta})\\) 是 \\(F(\boldsymbol{\theta})\\) 的海森矩阵。实际上，牛顿梯度法是在求原函数 \\(F(\boldsymbol{\theta})\\) 一阶导数的零点，而一阶导数的零点刚好对应原函数的极值点，关于牛顿迭代公式的导出，可以查看相应资料（如数值分析）。

&ensp;&ensp;&ensp;&ensp;我们以面阵为例，给出该条件下的表达式。首先设 \\(\boldsymbol{\theta}=(u,v)^T\\)，\\(u\\) 和 \\(v\\) 表示方位角和俯仰角的正（余）弦值。因此，我们将 \\((8)\\) 式改写为
$$
\begin{bmatrix}u \\\\\\\\ v\end{bmatrix}_{(\boldsymbol{\theta}^\star)}=
\begin{bmatrix}u \\\\\\\\ v\end{bmatrix}
 -\begin{bmatrix}\frac{\partial^2F}{\partial u^2} & \frac{\partial^2F}{\partial u\partial v} 
\\\\\\\\ \frac{\partial^2F}{\partial v\partial v}  & \frac{\partial^2F}{\partial v^2}\end{bmatrix}^{-1}
\begin{bmatrix}\frac{\partial F}{\partial u} \\\\\\\\ \frac{\partial F}{\partial v}\end{bmatrix} \tag{9}
$$

上式中，\\(\[\cdot\]_{(\boldsymbol{\theta}^\star)}\\) 表示新一轮的迭代值。在实际应用中，一般取初始值为波束指向 \\(\boldsymbol{\theta}_0\\)。

&ensp;&ensp;&ensp;&ensp;在本小节的最后，将直接给出式 \\((9)\\) 中各变量的表达式，详细的推导过程见参考文献。首先，给出 \\(F(\boldsymbol{\theta})\\) 的梯度表达式
$$\frac{\partial F}{\partial u}=2\left[\Re\left(\frac{\boldsymbol{d}_x^H\boldsymbol{z}}{\boldsymbol{w}^H\boldsymbol{z}}\right)-\mu_x\right] \tag{10}$$
$$\frac{\partial F}{\partial v}=2\left[\Re\left(\frac{\boldsymbol{d}_y^H\boldsymbol{z}}{\boldsymbol{w}^H\boldsymbol{z}}\right)-\mu_y\right] \tag{11}$$
上式中，\\(\boldsymbol{d}_x\\) 和 \\(\boldsymbol{d}_y\\) 分别为 \\(x\\) 方向和 \\(y\\) 方向上的差波束（假设面阵法向量为 \\(z\\) 轴方向），其相应的表达式为
$$\boldsymbol{d}_x=\frac{\boldsymbol{Q}^{-1}\partial\boldsymbol{a}/\partial u}{(\boldsymbol{a}^H\boldsymbol{Q}^{-1}\boldsymbol{a})^{1 / 2}}$$
$$\boldsymbol{d}_y=\frac{\boldsymbol{Q}^{-1}\partial\boldsymbol{a}/\partial v}{(\boldsymbol{a}^H\boldsymbol{Q}^{-1}\boldsymbol{a})^{1 / 2}}$$
式中，\\(\partial\boldsymbol{a}/\partial u\\) 和 \\(\partial\boldsymbol{a}/\partial u\\) 对应导向向量对方位角和俯仰角正（余）弦值的偏导数，其中面阵的导向向量为
$$\boldsymbol{a}(u,v)=\begin{bmatrix}\exp\[-j(2\pi/\lambda)(x_1u + y_1v) & \cdots & \exp\[-j(2\pi/\lambda)(x_Mu + y_Mv)\] \end{bmatrix}^T$$
式中， \\(M\\) 为阵元个数，由此可得
$$\left(\frac{\partial\boldsymbol{a}}{\partial u}\right)_i=-j(2\pi/x_i)\exp\[-j(2\pi/\lambda)(x_iu + y_iv)\]$$
$$\left(\frac{\partial\boldsymbol{a}}{\partial v}\right)_i=-j(2\pi/y_i)\exp\[-j(2\pi/\lambda)(x_iu + y_iv)\]$$
利用以上这些式子，可以求得差波束，而式 \\((10)\\) 和 \\((11)\\) 中，\\(\mu_x\\) 和 \\(\mu_y\\) 是两个方向上的偏差修正量，其公式如下
$$\mu_x=\Re\left[\frac{(\partial \boldsymbol{a}/\partial u)^H\boldsymbol{Q}^{-1}\boldsymbol{a}}{\boldsymbol{a}^H\boldsymbol{Q}^{-1}\boldsymbol{a}}\right]$$
$$\mu_y=\Re\left[\frac{(\partial \boldsymbol{a}/\partial v)^H\boldsymbol{Q}^{-1}\boldsymbol{a}}{\boldsymbol{a}^H\boldsymbol{Q}^{-1}\boldsymbol{a}}\right]$$
上式中，\\(\Re(\cdot)\\) 表示取实部。这样，\\(\nabla F(\boldsymbol{\theta})\\) 就可以由上文中这些式子求出。

&ensp;&ensp;&ensp;&ensp;接下来是海森矩阵的求解，海森矩阵求解的过程中有一些近似，用到了阵列的对称性，详情可以参见参考文献的附录部分。其对应的二阶偏导数如下
$$\frac{\partial^2 F}{\partial u^2}=2\mu_x^2 - 2\frac{\boldsymbol{d}_x^H\partial \boldsymbol{a}/\partial u}{\boldsymbol{w}^H\boldsymbol{a}} \tag{12}$$
$$\frac{\partial^2 F}{\partial v^2}=2\mu_y^2 - 2\frac{\boldsymbol{d}_y^H\partial \boldsymbol{a}/\partial v}{\boldsymbol{w}^H\boldsymbol{a}} \tag{13}$$
$$\frac{\partial^2 F}{\partial uv}=2\mu_x\mu_y - 2\frac{\Re\[\boldsymbol{d}_x^H\partial \boldsymbol{a}/\partial v\]}{\boldsymbol{w}^H\boldsymbol{a}} \tag{14}$$

&ensp;&ensp;&ensp;&ensp;利用式 \\((9)\\)-\\((14)\\)，我们就可以求出波达方向。

## 一个均匀线阵（ULA）的例子
&ensp;&ensp;&ensp;&ensp;若考虑一个均匀线阵，阵元个数 \\(M=8\\)，信噪比 \\(\text{SNR}=15\text{ dB}\\)，快拍数 \\(N=1000\\)。若假设波束指向 \\(\theta_0=20^\circ\\)，信号的实际波达方向为 \\(\theta=25^\circ\\)。首先考虑无干扰的情况，求解过程选用小批量随机梯度下降法（MGBD），将 1000 个样本划分为 100 个 Batch。其迭代结果如下


{{<center>}}
![](https://cdn.jsdelivr.net/gh/hopesAccount/Blog@master/Monopulse/Max_Likelihood/noJammer(MBGD).svg)
{{</center>}}


&ensp;&ensp;&ensp;&ensp;然后考虑干扰信号离波束指向较远的情况，若假设干扰信号位于 \\(\theta_j=10^\circ\\)，干噪比 \\(\text{JNR}=55\text{ dB}\\)，求解过程选用小批量随机梯度下降法（MGBD），将 1000 个样本划分为 100 个 Batch。其迭代结果如下


{{<center>}}
![](https://cdn.jsdelivr.net/gh/hopesAccount/Blog@master/Monopulse/Max_Likelihood/jammer=10(MBGD).svg)
{{</center>}}


&ensp;&ensp;&ensp;&ensp;接着考虑干扰靠近波束指向时的情况，若假设干扰信号位于 \\(\theta_j=18^\circ\\)，干噪比 \\(\text{JNR}=55\text{ dB}\\)，求解过程选用小批量随机梯度下降法（MGBD），将 1000 个样本划分为 100 个 Batch。其迭代结果如下


{{<center>}}
![](https://cdn.jsdelivr.net/gh/hopesAccount/Blog@master/Monopulse/Max_Likelihood/jammer=18(MBGD).svg)
{{</center>}}


## 结论
&ensp;&ensp;&ensp;&ensp;从这个均匀线阵的例子中我们可以看出，因为使用牛顿梯度法，该迭代过程收敛较快。

## 参考文献
U. Nickel, "Monopulse estimation with adaptive arrays," in IEE Proceedings F - Radar and Signal Processing, vol. 140, no. 5, pp. 303-308, Oct. 1993.