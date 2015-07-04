#Logistic Regression

1. No linear regression for classification

2. Decision Boundary is drawn from hypothesis function after fitting the $$\theta$$ parameters

3. Hypothesis Function:

   $$h(x) = \frac{1}{1+e^{-\theta^Tx}}$$

4. Cost Function:   convex is important->that is why we use log()
   
   $$Cost(h_{\theta}(x), y) =\frac{1}{m}\sum\limits_{i}^{m} (-y_ilog(h_\theta(x_i)) - (1-y_i)log(1-h_{\theta}(x_i))$$

   if $$y = 0$$:
   
   $$Cost(h_{\theta}(x), y) = -log(1-h_{\theta}(x))$$
   
   so, if $$h_{\theta}(x) = 0$$, the cost function will be close to zero; if $$h_{\theta}(x) = 1$$, the cost function will be go infinitiy.
   
   if $$y = 1$$:
   
   $$Cost(h_{\theta}(x), y) = -log(h_{\theta}(x))$$
   
   so if $$h_{\theta}(x) \sim 0$$, the cost function would be infinity while it will be zero if $$h_{\theta}(x) \sim 1$$
   
   if we plot the plot for y=1/0, its convex and intuitive.

3. Advanced Optimization instead of Gradient Descent
   - Gadient descent
   - Conjugate gradient
   - BFGS
   - L-BFGS

   Advantages:
   
   * No need to manually pick alpha
   * faster than GD(gradient descent)

   Disadvantage:
    
   * More complex

4. Multiple-Classification:
   One vs all method, make one against the remainder 
   max over i for $$\max\limits_{i}h_{\theta}^i(x)$$
   
5. Regulization: 
   - why? To deal with overfitting problemn(e.g.$$J_\theta = \frac{1}{2m}\sum\limits_{i=1}^m(h_\theta(x_i)- yi)^2+1000\theta_3^2+1000\theta_4^2$$))
   - Intuition: small values for all parameters;
       - "simpler" hypothesis;
       - less prone to overfitting
   - General form:
   $$J_\theta = \frac{1}{2m}\sum\limits_{i=1}^m((h_\theta(x_i)- yi)^2+\lambda\sum\limits_{i=1}^{m}\theta_{i}^2)$$(no $$\theta_0$$) where $$\lambda$$ is the regularization parameter, which stands for the tradoff between two goals:
   
      - fit the training data;
      - make paramters small(prevent overfitting);
   - Normal Equation:
   ![](http://i.imgur.com/64qtYj8.png)
       - invertibility:
   ![](http://i.imgur.com/Z2YQOzK.png)

6. Decision Boundary Plotting:
   - only 2 parameters: $$\theta^Tx = 0$$