# machine-learning-1-week-12-neural-networks-2-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 1 Week 12-Neural Networks 2 Solved](https://www.ankitcodinghub.com/product/machine-learning-1-week-12-neural-networks-2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98769&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 1 Week 12-Neural Networks 2 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Exercise 1: Neural Network Optimization

Consider the one-layer neural network

y = w⊤x + b

applied to data points x ∈ Rd, and where w ∈ Rd and b ∈ R are the parameters of the model. We consider

the optimization of the objective:

􏰍1 2􏰎 J(w)=Epˆ 2(1−y·t) ,

where the expectation is computed over an empirical approximation pˆ of the true joint distribution p(x, t) and t ∈ {−1,1}. The input data follows the distribution x ∼ N(μ,σ2I) where μ and σ2 are the mean and variance.

<ol>
<li>(a) &nbsp;Compute the Hessian of the objective function J at the current location w in the parameter space, and as a function of the parameters μ and σ of the data.</li>
<li>(b) &nbsp;Show that the condition number of the Hessian is given by: λ Exercise 2: Neural Network Regularization (10 + 10 + 10 P)
For a neural network to generalize from limited data, it is desirable to make it sufficiently invariant to small local variations. This can be done by limiting the gradient norm ∥∂f/∂x∥ for all x in the input domain. As the input domain can be high-dimensional, it is impractical to minimize the gradient norm directly. Instead, we can minimize an upper-bound of it that depends only on the model parameters.

We consider a two-layer neural network with d input neurons, h hidden neurons, and one output neuron. Let W be a weight matrix of size d×h, and (bj)hj=1 a collection of biases. We denote by Wi,: the ith row of the weight matrix and by W:,j its jth column. The neural network computes:

a =max(0,W⊤x+b) (layer1) j :,jj

f(x) = 􏰌j sjaj (layer 2)

where sj ∈ {−1, 1} are fixed parameters. The first layer detects patterns of the input data, and the second

layer computes a fixed linear combination of these detected patterns.
</li>
</ol>
(a) Show that the gradient norm of the network can be upper-bounded as:

</div>
</div>
<div class="layoutArea">
<div class="column">
√ 􏰑􏰑∂x􏰑􏰑≤ h·∥W∥F

<ol start="2">
<li>(b) &nbsp;Let ∥W∥Mix = 􏰒􏰌i ∥Wi,:∥21 be a l1/l2 mixed matrix norm. Show that the gradient norm of the network can be upper-bounded by it as:
􏰑∂f 􏰑

􏰑􏰑∂x􏰑􏰑 ≤ ∥W∥Mix
</li>
<li>(c) &nbsp;Show that the mixed norm provides a bound that is tighter than the one based on the Frobenius norm, i.e.</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
􏰑∂f 􏰑

</div>
</div>
<div class="layoutArea">
<div class="column">
λ1 ∥μ∥2

</div>
</div>
<div class="layoutArea">
<div class="column">
d

</div>
<div class="column">
= 1 + σ2 .

</div>
</div>
<div class="layoutArea">
<div class="column">
show that: .

Exercise 3: Programming (40 P)

</div>
<div class="column">
∥W∥Mix ≤

</div>
<div class="column">
√

</div>
<div class="column">
h·∥W∥F

</div>
</div>
<div class="layoutArea">
<div class="column">
Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 12 (programming) [WiSe 2021/22] Machine Learning 1

</div>
</div>
<div class="layoutArea">
<div class="column">
Neural Networks 2

In this homework, we will train neural networks on the Breast Cancer dataset. For this, we will use of the Pytorch library. We will also make use of scikit- learn for the ML baselines. A first part of the homework will analyze the parameters of the network before and after training. A second part of the homework will test some regularization penalties and their effect on the generalization error.

Breast Cancer Dataset

The following code extracts the Breast cancer dataset in a way that is already partitioned into training and test data. The data is normalized such that each dimension has mean 0 and variance 1. To test the robustness of our learning models, we also artificially inject 4% of mislabelings in the training data.

In [1]:

import utils

<pre>Xtrain,Ttrain,Xtest,Ttest = utils.breast_cancer()
</pre>
<pre>nx = Xtrain.shape[1]
nh = 100
</pre>
Neural Network Classifier

In this homework, we consider the same architecture as the one considered in Exercise 2 of the theoretical part. The class NeuralNetworkClassifier implementsthisnetwork.Thefunction reg isaregularizerwhichwesetinitiallytozero(i.e.noregularizer).Because

the dataset is small, the network can be optimized in batch mode, using the Adam optimizer.

In [2]:

import numpy,torch,sklearn,sklearn.metrics

from torch import nn,optim class NeuralNetworkClassifier:

def __init__(self):

<pre>        torch.manual_seed(0)
</pre>
self.model = nn.Sequential(nn.Linear(nx,nh),nn.ReLU())

with torch.no_grad(): list(self.model)[0].weight *= 0.1

self.s = torch.zeros([100]); self.s[:50] = 1; self.s[50:] = -1 self.pool = lambda y: y.matmul(self.s)

self.loss = lambda y,t: torch.clamp(1-y*t,min=0).mean()

def reg(self): return 0

def fit(self,X,T,nbit=10000):

<pre>        X = torch.Tensor(X)
        T = torch.Tensor(T)
</pre>
optimizer = optim.Adam(self.model.parameters(),lr=0.01) for _ in range(nbit):

<pre>            optimizer.zero_grad()
            (self.loss(self.pool(self.model(X)),T)+self.reg()).backward()
            optimizer.step()
</pre>
def predict(self,X):

return self.pool(self.model(torch.Tensor(X)))

def score(self,X,T):

Y = numpy.sign(self.predict(X).data.numpy()) return sklearn.metrics.accuracy_score(T,Y)

Neural Network Performance vs. Baselines

We compare the performance of the neural network on the Breast cancer data to two other classifiers: a random forest and a support vector classification model with RBF kernel. We use the scikit-learn implementation of these models, with their default parameters.

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
In [ ]:

from sklearn import ensemble,svm

<pre>rfc = ensemble.RandomForestClassifier(random_state=0)
rfc.fit(Xtrain,Ttrain)
</pre>
<pre>svc = svm.SVC()
svc.fit(Xtrain,Ttrain)
</pre>
<pre>nnc = NeuralNetworkClassifier()
nnc.fit(Xtrain,Ttrain)
</pre>
In [4]:

def pretty(name,model):

return ‘&gt; %10s | Train Acc: %6.3f | Test Acc: %6.3f’%(name,model.score(Xtrain,Ttrain),model.score(Xtest,Ttest

))

<pre>print(pretty('RForest',rfc))
print(pretty('SVC',svc))
print(pretty('NN',nnc))
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
&gt; &gt; &gt;

</div>
<div class="column">
<pre>RForest | Train Acc:  1.000 | Test Acc:  0.940
    SVC | Train Acc:  0.958 | Test Acc:  0.951
     NN | Train Acc:  1.000 | Test Acc:  0.884
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
The neural network performs not as good as the baselines. Most likely, the neural network has overfitted its decision boundary, in particular, on the mislabeled training examples.

Gradient, and Parameter Norms (25 P)

For the model to generalize better, we assume that the gradient of the decision function should be prevented from becoming too large. Because the gradient can only be evaluated on the current data distribution (and may not generalize outside the data), we resort to the following inequality we have proven in the theoretical section for this class of neural network models:

$$ \text{Grad} \leq \|W\|_\text{Mix} \leq \sqrt{h}\|W\|_\text{Frob} $$

where

$\|W\|_\text{Frob} = \sqrt{\sum_{i=1}^d \sum_{j=1}^h w_{ij}^2}$

$\|W\|_\text{Mix} = \sqrt{\sum_{i=1}^d \big(\sum_{j=1}^h | w_{ij}|\big)^2}$

$\text{Grad} = \textstyle \frac1N \sum_{n=1}^N\|\nabla_{\boldsymbol{x}}f (\boldsymbol{x_n})\|$

and where $d$ is the number of input features, $h$ is the number of neurons in the hidden layer, and $W$ is the matrix of weights in the first layer (Note that in PyTorch, the matrix of weights is given in transposed form).

Asafirststep,wewouldliketokeeptrackofthesequantitiesduringtraining.Thefunction Frob(nn) thatcomputes$\|W\|_\text{Frob}$isalready implemented for you.

Tasks:

Implement the function Mix(nn) that receives the neural network as input and returns $\|W\|_\text{Mix}$.

Implement the function Grad(nn,X) that receives the neural network and some dataset as input, and computes the averaged gradient norm ($\text{Grad}$).

In [5]:

def Frob(nn):

W = list(nn.model)[0].weight return (W**2).sum()**.5

def Mix(nn):

# ———————————-

# TODO: Replace by your code

# ———————————- import solution; return solution.Mix(nn) # ———————————-

def Grad(nn,X):

# ———————————-

# TODO: Replace by your code

# ———————————-

import solution; return solution.Grad(nn,X) # ———————————-

</div>
</div>
<div class="layoutArea">
<div class="column">
The following code measures these three quantities before and after training the model.

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
In [6]:

def fullpretty(name,nn):

return pretty(name,nn) + ‘ | Grad: %7.3f | WMix: %7.3f | sqrt(h)*WFrob: %7.3f’%(Grad(nn,Xtest),Mix(nn),nh**.5

</div>
</div>
<div class="layoutArea">
<div class="column">
*Frob(nn))

<pre>nnr = NeuralNetworkClassifier()
print(fullpretty('Before',nnr))
nnr.fit(Xtrain,Ttrain)
print(fullpretty('After',nnr))
</pre>
<pre>&gt;     Before | Train Acc:  0.391 | Test Acc:  0.372 | Grad:
5.751
</pre>
</div>
<div class="column">
<pre>0.389 | WMix:
7.297 | WMix:  40.103 | sqrt(h)*WFrob:
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
&gt; 56.739

</div>
<div class="column">
<pre>After | Train Acc:  1.000 | Test Acc:  0.884 | Grad:
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the inequality $\text{Grad} \leq \|W\|_\text{Mix} \leq \sqrt{h} \|W\|_\text{Frob}$ we have proven also holds empirically. We also observe that these quantities tend to increase as training proceeds. This is a typical behavior, as the network starts rather simple and becomes complex as more and more variations in the training data are being captured.

Norm Penalties (15 P)

We consider the new objective $J^\text{Frob}(\theta) = \text{MSE}(\theta) + \lambda \cdot (\sqrt{h} \|W\|_\text{Frob})^2$, where the first term is the original mean square error objective and where the second term is the added penalty. We hardcode the penalty coeffecient to $\lambda = 0.005$. In principle, for maximum performance and fair comparison between the methods, several of them should be tried (also for other models), and selected based on some validation set. Here, for simplicity, we omit this step.

A downside of the Frobenius norm is that it is not a very tight upper bound of the gradient, that is, penalizing it is does not penalize specifically high gradient. Instead, other useful properties of the model could be negatively affected by it. Therefore, we also experiment with the mixed-norm regularizer $\textstyle \lambda \cdot \|W\|_\text{Mix}^2$, which is a tighter bound of the gradient, and where we also hardcode the penalty coefficient to $\lambda = 0.025$.

Task:

Create two new classifiers by reimplementing the regularization function with the Frobenius norm regularizer and Mixed norm regularizer respectively. You may for this task call the norm functions implemented in the question above, but this time you also need to ensure that these functions can be differentiated w.r.t. the weight parameters.

The code below implements and train neural networks with the new regularizers, and compares the performance with the previous models.

In [7]:

import solution

class FrobClassifier(NeuralNetworkClassifier):

def reg(self):

# ———————————-

# TODO: Replace by your code

# ———————————-

import solution; return solution.FrobReg(self) # ———————————-

class MixClassifier(NeuralNetworkClassifier):

def reg(self):

# ———————————-

# TODO: Replace by your code

# ———————————-

import solution; return solution.MixReg(self) # ———————————-

In [8]:

<pre>nnfrob = FrobClassifier()
nnfrob.fit(Xtrain,Ttrain)
</pre>
<pre>nnmix = MixClassifier()
nnmix.fit(Xtrain,Ttrain)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>4.966 | sqrt(h)*WFrob:
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
In [9]:

<pre>print(pretty('RForest',rfc))
print(pretty('SVC',svc))
print(fullpretty('NN',nnc))
print(fullpretty('NN+Frob',nnfrob))
print(fullpretty('NN+Mix',nnmix))
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>&gt;
&gt;
&gt;
56.739
&gt;    NN+Frob | Train Acc:  0.961 | Test Acc:  0.954 | Grad:
2.689
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>RForest | Train Acc:  1.000 | Test Acc:  0.940
    SVC | Train Acc:  0.958 | Test Acc:  0.951
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>NN | Train Acc:  1.000 | Test Acc:  0.884 | Grad:
</pre>
</div>
<div class="column">
<pre>7.297 | WMix:  40.103 | sqrt(h)*WFrob:
0.735 | WMix:   1.767 | sqrt(h)*WFrob:
0.745 | WMix:   1.637 | sqrt(h)*WFrob:
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>&gt;     NN+Mix | Train Acc:  0.951 | Test Acc:  0.961 | Grad:
4.138
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the regularized neural networks now performs on par with the baselines. It is interesting to observe that the mixed norm penalty more selectively reduced the gradient, and has let the Frobenius norm take higher values.

</div>
</div>
</div>
</div>
<div class="page" title="Page 6"></div>
<div class="page" title="Page 7"></div>
