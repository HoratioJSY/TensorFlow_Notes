## Jacobian 矩阵与 Hessian 矩阵

第四章数值计算，基于梯度的最优化方法

- Jacobian 矩阵定义：如果我们有一个函数 $f​$ : $\mathbb{R}^m\rightarrow\mathbb{R}^n​$，$f​$ 的 Jacobian 矩阵 $J\in\mathbb{R}^{n\times m}​$ 定义为 $J_{i, j}=\frac{\partial}{\partial x_j}f(x)_i​$。
- Hessian 矩阵定义：若存在一个函数 $f$ : $\mathbb{R}^m\rightarrow\mathbb{R}$，$f$ 的一阶导数（关于 $x_j$ ）关于 $x_i$ 的导数记为 $\frac{\partial^2}{\partial x_i \partial x_j}f$，那么 Hessian 矩阵 $H(f)(x)$ 定义为 $H(f)(x)_{i,j}=\frac{\partial^2}{\partial x_i \partial x_j}f(x)$。

这里我们可以探讨两个问题，首先 Jacobian 矩阵在深度学习中的应用是什么。因为我们知道 Hessian 矩阵一般应用在牛顿法等涉及二阶导数的最优化算法中，或者是用于极值点的二阶导数测试，但是我不太了解 Jacobian 矩阵在 DL 中到底有什么应用。其次，二阶导数可以视为对曲率的衡量，那我们应该怎样理解这种衡量。



### Jacobian 矩阵的应用

若 $f(x)$ 为 m 维函数向量，即 $f(x)=[f_1(x), f_2(x), \cdots,f_m(x)]^T$，且其中 $x$ 为 n 维变量向量，那么 $f(x)$ 的 Jacobian 矩阵可以表示为：

$\nabla\mathbf f(\mathbf x)=\begin{bmatrix}\frac{\partial f_1(\mathbf x)}{\partial x_1} & \frac{\partial f_1(\mathbf x)}{\partial x_2} &  \cdots & \frac{\partial f_1(\mathbf x)}{\partial x_n} \\ \frac{\partial f_2(\mathbf x)}{\partial x_1} &  \frac{\partial f_2(\mathbf x)}{\partial x_2}& \cdots & \frac{\partial f_2(\mathbf x)}{\partial x_n}\\ \cdots & \cdots &  \cdots& \cdots\\ \frac{\partial f_m(\mathbf x)}{\partial x_1} &  \frac{\partial f_m(\mathbf x)}{\partial x_2} & \cdots & \frac{\partial f_m(\mathbf x)}{\partial x_n}\end{bmatrix}_{m \times n} = \begin{bmatrix}g_1(\mathbf{x})^T \\ g_2(\mathbf{x})^T \\ \cdots \\ g_m(\mathbf{x})^T  \end{bmatrix} $



 根据目前的了解，Jacobian 矩阵在反向传播中有很广泛的应用。因为深层网络在可以视为一系列的复合函数，而复合函数的求导可以很自然地根据链式法则一层层地求，且形式为偏导运算的累乘。又因为损失函数对每层权重的梯度可以视为一个 Jacobian 矩阵，因此损失函数对某层权重的梯度可以表示为 Jacobian 矩阵的乘积。

以FCN为例，三层网络的推断可以表示为 $\hat{y}=f(f(x))$，损失函数表示为 $L(\hat{y})$。那么损失函数对预测值的导数为 $\frac{\partial L}{\hat{y}}$，同理第二层与第一层权重的梯度分别为 $\frac{\partial L}{\hat{y}}\,\frac{\partial \hat{y}}{\partial f(x)}$ 和 $\frac{\partial L}{\hat{y}}\,\frac{\partial \hat{y}}{\partial f(x)} \, \frac{\partial f(x)}{\partial x}$。若 $\nabla_{\mathbf{x}}\mathbf{f(x)}$ 表示向量 $\mathbf{f(x)}$ 的 Jacobian 矩阵，那么第二层与第一层的权重梯度分别可以写为 $\nabla_{\mathbf{\hat{y}}}L\ \nabla_{\mathbf{f(x)}}\mathbf{\hat{y}}$ 和 $\nabla_{\mathbf{\hat{y}}}L\ \nabla_{\mathbf{f(x)}}\mathbf{\hat{y}}\ \nabla_{\mathbf{x}}\mathbf{f(x)}$。以第一层权重的梯度为例，它可以表示为 Jacobian 矩阵的乘积，且维度可以表示为：

$\nabla_{x}L=\begin{bmatrix} \cdots & \cdots\end{bmatrix} \times \begin{bmatrix} \cdots & \cdots \\ \cdots & \dots \end{bmatrix} \times \begin{bmatrix} \cdots & \cdots \\ \cdots & \dots \end{bmatrix}$

这种链式矩阵乘法需要确定计算顺序以提高计算效率，这里因为行向量左乘矩阵得行向量，所以直接从左到右计算可以很高效。

Notes：如果扩展成文的话，可以从实践中的 Jacobian 矩阵如何应用到BP算法出发。例如如何存储到 grad_table，在符号到数值的微分（Caffe）和采用计算图的微分（TF）是否也需要使用 Jacobian 计算，矩阵中的元素如何更新的等。



### Hessian 矩阵

Hessian 矩阵常用于牛顿法，并提供二阶导数。这一部分讨论为什么二阶导能给梯度方向提供更好的下降方向，以及为什么二阶导可以视为曲率的衡量。

二阶导提供更好下降方向的原因在DL书中描述得比较明白，我们可以通过泰勒展开式（展开到二阶）预期一个梯度下降步能表现有多好。具体的，若我们在 $x_1$ 点展开展开 $f(x)$ 的二阶泰勒级数：

$f(x)\approx f(x_1)\ +\ (x-x_1)^Tg\ +\ \frac12(x-x_1)^TH\,(x-x_1)$

其中 $x$ 与 $x_1$ 都为列向量，$g$ 为梯度向量，$H$ 为 $x_1$ 点的 Hessian 矩阵。若更新的点 $x_2=x_1-\epsilon g$ ，代入可得：

$f(x_1-\epsilon g)\approx f(x_1)\ -\ \epsilon g^Tg\ +\ \frac12\epsilon^2g^THg$

其中表达式右边第一项表示函数的原始值，第二项表示函数斜率产生的预期改善，第三项表示函数曲率产生的矫正。按照泰勒展开式的性质，后面展开的项对函数值的贡献越来越小，因此一阶导提供的信息量最大，二阶导提供的信息次之。在梯度下降中，若我们令上式 $f(x)=0$ ，那么二阶导能为估计最优解提供有用的信息，或修正下降方向。

上式第三项表示函数曲率对估计的矫正，为什么？

根据高数，曲率的定义是一定转角微分与对应弧长微分的比值，即 $K=|\frac{d\alpha}{ds}|=\frac{|y''|}{(1+y'^2)^{\frac32}}$。若在给定梯度下，二阶导可以视为曲率的衡量。

又因为 Hessian 矩阵是实对称的（满足 $H_{i,j}=H_{j,i}$ ），我们可以将其分解为一组实特征值和一组特征向量的正交基。而分解后的特征值表示主曲率的大小，即在该点不同方向的弯曲程度。因为梯度衡量函数在某个点的变化趋势，而二阶导衡量了梯度在某个点的变化趋势，即二阶导或弯曲程度衡量了函数的变化速度。例如在某一点，二阶导大于零说明梯度递增，而梯度大于零只能说明函数递增，若加上梯度递增这一信息，那么函数可以判断将变得更加陡峭或变化快速。（感觉好像速度与加速度的关系。。。这一部分理解不充分）