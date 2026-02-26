<details markdown="block">
<summary> Computing gradients with PyTorch </summary>

## Computing gradients with PyTorch

Video suggestion: [Backpropagation by Patrick Loeber](https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4). The author explains what is a computation graph and how PyTorch uses it to compute gradients. The example uses a very simple model: $\hat{y}=w \cdot x$ and the MSE loss which is just $(\hat{y}-y)\^2$.

- Pipeline for the regression problem:
  - Prepare data
  - Design model (input, output size, model: perceptron)
  - Construct loss and optimizer
  - Training loop
    - Forward pass: prediction and loss
    - Backward pass: gradients
    - Update weights

In the following examples, one starts with a step by step code in Python for the linear regression model  that is based on our knowledge of  the gradient expression for the MSE loss function and we convert it into a Pytorch code that can be easily generalized to other models and other loss functions.

Video suggestion: [Gradient Descent with Autograd and Backpropagation by Patrick Loeber](https://www.youtube.com/watch?v=E-I2DNVzQLg). The author first uses `numpy` to create a gradient descent script for a  linear regression model and then replaces the manual gradient calculation by `PyTorch` automatic gradient calculation. The following scripts illustrate the possibilities to move from a low level Python code to a higher level PyTorch code for the simple Linear Regression problem. The higher-level code can be easily adapted to more complex models and other optimizers.

- [numpy version](https://github.com/patrickloeber/pytorchTutorial/blob/master/05_1_gradientdescent_manually.py)
- [torch version](https://github.com/patrickloeber/pytorchTutorial/blob/master/05_2_gradientdescent_auto.py)
- [torch version with torch loss criterion and optimizer](https://github.com/patrickloeber/pytorchTutorial/blob/master/07_linear_regression.py)

The following table illustrates the changes from a basic Python script which is dependent on the model, loss, etc,  to a PyTorch higher-level script that can easily generalized to other models, loss functions and optimizer strategies.

| Basic Python | PyTorch 
|---|---
| Define model explicitly | Use a pre-defined model
|`def predict(x):`|`torch.nn.Linear(in_size,out_size)`
| Define loss explicitly | Use a pre-defined loss function
|`def loss(y,y_pred):`|`loss=torch.nn.MSEloss(y,y_pred)`
| Loss optimization strategy | Use a pre-defined optimizer
| | `optimizer=torch.optim.SGD(params, learn_rate)`
| Compute *ad hoc* gradient | **Use built-in backpropagation mechanism**
|`def gradient(x,y,y_pred):`|`loss.backward()`
|Update weights explicitly| `optimizer.step()`

---

</details>
<details markdown="block">
<summary> Exercise with pseudo-code for SGD </summary>

## Exercise with pseudo-code for SGD

Consider the following pseudo-code to train a simple Linear Regression model. What is the *loss* function that we aim at minimizing? What is the strategy to reduce the *loss* in each iteration? Is there a risk of *over-fitting*?
  
  ---
  Pseudo code for SGD (stochastic gradient descent) to fit a linear regression:
  
  - Dataset:  $D = {(x_1^{(i)}, ..., x_n^{(i)}, y^{(i)})}\_{i=1}\^N$  `N observations, n features`
  - Learning rate:  $\eta$ `Small positive value`
  - Max iterations: max_iter `Number of epochs`
  - Initial weights $w$ := $(w_0, w_1, ..., w_n)$ `Typically, all zero`
  - For iter := 1 to max_iter 
    - For each  $(x_1, ..., x_n, y) \in D$  `Update weights after each example`
      - $\hat{y}$ := $w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$ `Predict response with current weights`
      - error := $y-\hat{y}$
      - $w_0$ := $w_0 + \eta \cdot$ error # `Update weight (bias)`
      - For $j$ := 1 to $n$
        - $w_j$ := $w_j + \eta \cdot$ error $\cdot x_j$ # `Update weight (for each feature)`
          
  ---

- Create a `LinearRegression` class with a `fit` method to implement the pseudo code above. Add to your class a `predict` method to make new predictions using the fitted model. Test your class with the following example.
    
  ```Python
  # Create synthetic data
  np.random.seed(0)
  X = np.random.rand(100, 1) # array with 100 rows and 1 column (1 feature)
  y = 2 + 3 * X + np.random.randn(100, 1) * 0.1
  # Create and train the model
  model = LinearRegression(learning_rate=0.1, max_iter=1000)
  model.fit(X, y)
  # Make predictions
  X_test = np.array([[0.5]])
  y_pred = model.predict(X_test)
  print(f"Prediction for X=0.5: {y_pred[0]}")
  ```
- Create an animation that shows the position of the fitted line for successive epochs for the example above.
- How can you adapt the code to address a classification problem where the response $y$ can only be 0 or 1?

</details>

<!---

Below, we discuss a `PyTroch` gradient descent script for the linear regression problem, and we compare the result with the optimal coefficients obtained by *least squares*. The code below shows how *training loss* is  computed.

The most specific part of the algorithm is the gradient computation. Note that the *gradient machinery* of `PyTorch` is turned-on for each weight with `requires_grad = True` as in the following case:

    coeffs=torch.tensor([-20.,-10.]).requires_grad_()

Then, the derivatives can be computed for any continuous function of the weights in tensor `coeffs`. In particular, the *loss* $L$ is defined as a function (that can be arbitrarily complicated) of the weights, and the *gradient* $\nabla L({\rm \bf w}^{\star})$ for the current set of weights ${\rm \bf w}^{\star}$ is computed with

    loss.backward()

Finally, the weights are updated with

    coeffs.sub_(coeffs.grad * step_size)

where method `sub_` is substraction for weight updating ${\rm \bf w}^{\star}:={\rm \bf w}^{\star} - \eta \\, \nabla L({\rm \bf w}^{\star})$, and  the learning rate $\eta$ is called `step_size` in the code.

PyTorch accumulates the gradients on subsequent backward passes, i.e. it accumulates the gradients on every `loss.backward()` call. Since  the update is to be based on the current gradient value, we need to include the `coeffs.grad.zero_()` instruction to zero the gradients before the next pass.

Try changing the learning rate to see what is the result (try for instance `step_size=0.1`).

<details>
  <summary>Script: gradient descent with PyTorch, train only, stochastic gradient descent</summary>

```python
# This example illustrates: gradient descent with PyTorch, train only, stochastic gradient descent (SGD)
import matplotlib.pyplot as plt
import torch
import numpy as np
torch.manual_seed(42)

step_size = 0.001  # learning rate
iter = 20  # number epochs

############################################ Creating synthetic data
# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)  # view converts to rank-2 tensor with one column
func = -5 * X + 2
# Adding Gaussian noise to the function f(X) and saving it in Y
y = func + 0.4 * torch.randn(X.size())

########################################## Baseline: Linear regression LS solution
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
print('Least square LR coefficients:',reg.intercept_,reg.coef_)

####################################################### Gradient Descent
# initial weights
coeffs = torch.tensor([-20., -10.], requires_grad=True)

# defining the function for prediction (linear regression)
def calc_preds(x):
    return coeffs[0] + coeffs[1] * x

# Computing MSE loss for one example
def calc_loss_from_labels(y_pred, y):
    return torch.mean((y_pred - y) ** 2) # MSE

# lists to store losses for each epoch
training_losses = []

# epochs
for i in range(iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = calc_preds(X)
    loss = calc_loss_from_labels(y_pred, y)
    training_losses.append(loss.item())

    # Stochastic Gradient Descent (SGD): update weights 
    for j in range(X.shape[0]):
        # randomly select a data point
        idx = np.random.randint(X.shape[0])
        x_point = X[idx]
        y_point = y[idx]

        # making a prediction in forward pass
        y_pred = calc_preds(x_point)

        # calculating the loss between predicted and actual values
        loss = calc_loss_from_labels(y_pred, y_point)

        # compute gradient
        loss.backward()

        # update coeffs
        with torch.no_grad():
            coeffs.sub_(coeffs.grad * step_size)
            # zero gradients
            coeffs.grad.zero_() # prevents from accumulating

print('coeffs found by stochastic gradient descent:', coeffs.detach().numpy())

# plot training loss along epochs
plt.plot(training_losses, '-g')
plt.xlabel('epoch')
plt.ylabel('loss (MSE)')
plt.show()
```
</details>

--->
