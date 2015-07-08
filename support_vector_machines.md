# Support Vector Machines
1. Optimization Objective:

    - Definition:
    $$\min\limits_{\theta}C\sum\limits_{i=1}^{m}y^icost_1(\theta^Tx^i)+(1-y^i)cost_0(\theta^Tx^i) + \frac{\lambda}{2}\sum\limits_{i=1}^{n}\theta_i^2$$;
    - Hypothesis: $$h_{\theta}(x) = 1/0 if(\theta^Tx\geq 0$$ or otherwise);
2. Large Margin Classifier(SVM)(C is very large, emphasize the first term in cost function, may overfit for outlier)
    - how to represent the decision boundary on two features plots(linearly separatable)
    ![](http://i.imgur.com/k172HG8.png)
    - why this optimization leads to large margin classifier?
3. MATH behind tlarge margin classification
    - 