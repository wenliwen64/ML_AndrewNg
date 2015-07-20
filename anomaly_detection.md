# Anomaly Detection
1. Density Estimation:
     - Anomaly detection:
         - Problem Motivation: Aircraft engine features example$${x^1, x^2, x^3....}$$, Is $$x_{test}$$ ok or not? Or fraud detection, xi = features of user i;s activities. Model p(x)(probability) from data, identify unusual users by check ing which have p(x)<epsilon; Monitoring computers in a data center: memory use, CPU load, CPUload/network_traffic, number of disk accesses/sec; Manufacturing
         - Gaussian Distribution: $$x\sim N(\mu, \sigma^2)$$, $$p(x;\mu, \sigma^2)=\frac{2}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
         - Anomaly Detection Algorithms:
             - Training set: $${x^1, x^2,....x^m}, x^i\in\mathbb{R}^n$$
             - $$P(x) = p(x_1; \mu_1, \sigma^2_1)p(x_2...)=\prod p(x_i;\mu_i,\sigma_i^2)$$ those parameters can be estimated by $$\mu_j = \frac{1}{m}\sum\limits_{i=1}^m x_j^i, \sigma_j^2=\frac{1}{m}\sum\limits_{i=1}^m (x^i_j-\mu_j)^2$$
             - It is a anomaly if $$P(x)<\epsilon$$;
     
2. Develope and evaluate an anomaly detection system:
    - assume we have some labeled data, of anomalous and non-anomalous examples(y = 0 if normal, y = 1 if anomalous)
    - trining set: $$x^1, x^2, x^3...$$
    - cross validation set: 
    - test set:
    - example: 10000 good engines, 20 flawed engines, training set: 6000 good engines, cv: 2000 good 10 flawed, test: 2000 good 10 flawed; data is very skewed
    - algo evaluation:1.fit p(x),2. on a cv /test example, predict based on $$\epsilon$$
3. Anomaly Detection vs. Supervised Learning
   - very small number of positive examples(y=1), large number of negatives->anomaly detection -> many different types of anomalies, hard  for any algorithms to learn from positive examples, future anomalies look nothing like any of the anomalous examples
   - large number of positive AND negative examples, enough positive examples for algo to get a sense of what postive examples are like: email spam claasification, weather prediction, cancer classification
4. Choosing what features to use:
    - non-gaussian features - transform it
    - error analysis for anomaly detection:
    - monitoring computers in a data center:
        - x1= memory use, x2 = number of disk accesses/sec, x3 = cpu load, x4 = networktraffic, x5 = cpu/network, x6 = cpu^2/network...
5. Multi-variate Gaussian Distribution
    - Formulation: $$x\in\mathbb{R}^n$$, dont model p(x_1), p(x_2), ...separately, model p(x) all in one go. $$P(x;\mu,\Sigma) = \frac{1}{\sqrt{2\pi}^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$$
    - positive or negative correlation observed from sigma matrix(off-diagonal elements)
6. Anomaly Detection using the Multivariate Gaussian Distribution
    - ![](http://i.imgur.com/lR6fVUA.png)
    - original model corresponds to a special case of multivariate gaussian distribution where off diagonal elements are zero is sigma matrix
    - so multivariate gaussian can automatically capture the correlations between features. but original is computationally cheaper(n = 10000),must have m > n , or else sigma is non-invertible. original one has no this restriction. Redundant features may make sigma non-invertible. 2 solutions to non-invertible sigma(a. m >= 10n, b.find redundant features) 
  
       