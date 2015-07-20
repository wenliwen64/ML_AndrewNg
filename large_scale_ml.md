# Large scale machine learning
1. It is not who has the best algorithm that wins, it is who has the most data.
2. Learning with large datasets: plot error vs. training set size(J_train, J_cv)
3. Stochastic Gradient Descent:
    - Gradient descent is computationally expensive(batch gradient descent);
    - stochastic gradient descent:

    $$cost(\theta, (x^i, y^i))= 0.5\times(h_\theta(x^i) - y^i)^2$$
    
    $$J_{train}(\theta) = 1/m \sum\limits_{i=1}^mcost(\theta, (x^i, y^i))$$
    
    First of all, randomly shuffle dataset:
    
    ```matlab
    pseudocode
    for i = 1:m
        theta(j) = theta(j) - alpha * (h_theta(x(i))-y(i))*x(i)
    end
    ```
    $$\theta_j = \theta_j - \alpha\frac{\partial}{\partial \theta_j}cost(\theta, (x^i, y^i))$$
  
     Usually iterate 1-10 times
     
 4. Mini-Batch Gradient Descent:
     - batch gradient descent: all of m examples
     - stochastic gradient descent: 1 example in each iteration
     - mini-batch gradient descent: b examples in each iteration(m = 10, 2-200)
     - why mini-batch better over stochastic: parallelize computation, if vectorization is well done??

 5. Stochastic gradient descent covergence:
     - every 1000 iterations, plot cost(\ehta,(x^i, y^i)) averaged over the last 1000 examples processed by algorithm.
     
    ![](http://i.imgur.com/QmO1VGt.png)
    - the first graph red mean smaller alpha, second graph, red means average over 5000 examples, third one is just not working for two different
    - Keeping alpha constant is pretty common stratedgy or u can gradully decrease its value so you can get better parameters for global minimum
 6. Online Learning:
     - online learning setting: shipping service website where user comes , specifies origin and destinations, you offer to ship their package for some asking price,  and users sometimes choose to use your service sometimes not.
     - Features x capture properties of user, of origin/destination and asking price, We want to learn $$p(y = 1|x;\theta)$$ to optimize price (using logistic regression)
     
    Repeat forever{
    
       Get(x, y ) corresponding to user;
       Update $$\theta$$ using(x,y) `theta(j) = theta(j) - alpha*x(j)*(h_theta(x) - y)` j = 0-n
       
    }
    
    - predict CTR(click through rate) and show the 10 phones they are mostly likely to click on, like sell the phone (match search queries)
7. Mapreduce and Data Parallelism
    - Batch gradient descent: machine1, use first 100 examples;
    - $$temp_j^1 = \sum\limits_{i=1}^{100}(h_\theta(x^i) - y^i)x_j^i$$...
    - $$\theta_j = \theta_j - \alpha\sum(h_\theta(x^i) - y^i)x^i_j$$ by combing results from all of the computers
    - multi-core machines can also been used

