#Learn Neural Network(for classification problems)

1. Cost function:
    - L = total no. of layers in network
    - $$s_l$$ = no. of units in layer 1(not counting bias unit)
    - Binary classification(y = 0 or 1):
    - Multi-class classification(K classes, k dimensional vector, k >= 3): 
    
    $$J(\Theta) = -\frac{1}{m}[\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{K}y_k^{i}log(h_\Theta(x^{(i)}))_k+(1-y_k^{i})log(1-h_\Theta(x^{(i)}))_k]+\frac{\lambda}{2m}\sum\limits_{l=1}^{L-1}\sum\limits_{j=1}^{s_{l+1}}\sum\limits_{i=1}^{s_l}(\Theta_{ji}^{(l)})^2$$
    
    
2. Back-propagation:
   - Intuition: $$\delta_j^l = (\Theta^l)^T\delta^{l+1}.*g'(z^l)$$, where $$g'(z^l)$$ is the derivative with respect to $$z$$, which equals $$a^l.*(1-a^l)$$
   - Good to know: if $$\lambda = 0$$,  $$\frac{\partial J(\Theta)}{\partial \Theta_{ji}^l} = a_j^l\delta_i^{l+1}$$
   - Algorithms:
       - Set $$\Delta_{ji}^l = 0$$
       - ![](http://i.imgur.com/ddcp6Cf.png)
  - Intuition: ![](http://i.imgur.com/l9NheY1.png)

3. Implementation tips:
  - Unroll and reshape a matrix:
  - Gradient Checking: 1. numberical compute gradient; 2. check gradApprox ~ DVec(from backpropagation；
  - Notes:
      - Implement backprop to compute DVec(Unrolled $$D^1, D^2, D^3$$);
      - Implement numerical gradient check to compute gradApprox;
      - Make sure they give similar values;
      - Turn of gradApprox(slower than DVec);
            
4. Random Initialization:
![](http://i.imgur.com/Qt02umS.png)
    - if use zeros(n, 1) to initialize the theta, it would end up with all same delta;
    - Symmetry Breaking: Initialize each $$\Theta_{ij}^l$$ to a random value in $$[-\epsilon, \epsilon]$$: `Theta1 = rand(10,11)*2*init_epsilon - init_epsilon;`
5. Work flow:
   - Training a neural netwrok: no. of input units; no. of output units; reasonabel default: 1 hidden or larger, every layer has same units(usually more the better);
   - flow: 
       - Rndomly initialize weights;
       - Implement forward prop to get $$h_{\Theta}(x^i)$$
       - Implement code to compute cost function $$J(\Theta)$$
       - Implement backprop to compute partial derivatives $$\frac{\partial J(\Theta)}{\partial \Theta}$$
       - Use gradient chacking to check
       - Use gradient descent or advanced optimization medhotd with backprop to minimize $$J(\Theta)$$ as function of $$\Theta$$(if $$J(\Theta)$$ is non-convex, may stuck in local minimum)

