# Clustering
---
1. Unsupervised learning
    - unlabeled training set: Training Set := $$\{x^1, x^2, x^3, x^4...\}$$
          - market segmentationm
          - social network analysis
          - organize computing clusters
          - anstronomical data analysis
2. K-Means Algorithms(most famous algo of clustering)
    - cluster centroids(randomly initialized)
        - if you want to split it into 2 groups(two centroids) and computer the distance from those centroids;
        - move centroids to means in each group;
        - Definition:
            - Input: k(number of clusters); training set {x1, x2, x3, ...}
            - 1.Randomly initialize K cluster centroids $$\mu_1, \mu_2,...\in\mathbb{R}^n$$ where n is the dimension of each data point(ignore the $$x_0^i = 1$$)
            - 2.Repeat
           
           ```matlab
           % puseudocode
           for i = 1:m
               c(i) =  index(from 1 to K) of centroids closeset to x(i)
           end
           for k = 1:K
               mu(k) = means(points in cluster k)% |x-x|^2
           end
           ```
           - K-means for non-separated clusters(Tshirt size: height/weight, cluster data to determine the population of each size)
           - Optimization objective:
               - $$c^i$$ = index of cluster to which example $$x^i$$ is currently assigned
               - $$\mu_k$$ = cluster centroid k($$\mu_k\in\mathbb{R}^n$$)
               - $$\mu_{c^i}$$ = cluster centroid of cluster to which example $$x^i$$ has been assigned
               - Cost function: $$J(c^1, c^2...c^m, \mu_1, \mu_2...\mu_K) = \frac{1}{m}\sum\limits_{i=1}^{m}||x^i - \mu_{c^i}||^2$$ will always decrease with increasing iteration loops.

               
2. How to intialize K-means?
    - Should have K < m
    - Randomly pick K training examples
    - Set $$\mu_1, ....\mu_K$$ equal to these K examples, randomize the indices
    - Local optima issue;
    - Random initialization:
    
    ```matlab
       for i = 1: 100
           randomly initializae K-means
           run K-means, get c(1), c(2)....., mu(1), mu(2)...,mu(k)
           compute cost function
       end
       pick the clustering that gave you the minimum cost function
    ```
3. How to choose the number of Clusters
    - What is the right value of K?
        - Elbow method: J vs. K(no. of clusters)->choose the elbow point;sometimes its hard to distinguish the elbow
        - based on a metric for how well it performs for that later purpose(like tshirt sizes, small, medium, large, OR xs, s, m, l, xl);
4. UL Motivation I: Data compression
   - Reduce data from 2D to 1D: x1 and x2 is very linear, project the data points to fitting straight lines(z1):$$x^1\in\mathbb{R}^2 \rightarrow z^1 \in\mathbb{R}...$$, make learning more quickly;
   - Reduce data from 3D to 2D(1000D ->100D): $$x^i\in\mathbb{R}^3$$ project these data to a plane and assign them z1 and z2 from the coordinates from thsi projected plane;
 5. UL Motivation II: Data Visularization
     - 2D/3D
 6. Principle Component Analysis:
     - Problem Formulation: Reduce from 2D to 1D: find a direction onto which to project the data so as to minimize the projection error //
     - Reduce from nD to kD :...... to minimize the projection error;= lower dimensional surface;
     - Difference from linear regression: vertical distance or distance from that straight line
 7. PCA Algorithm:
     - Data preprocessing: 1. mean normalization; 
     - Compute "covariance matrix": $$\Sigma = \frac{1}{m}\sum\limits_{i=1}^{m}(x^i)(x^i)^T$$(x is n by 1 vector)
     - cmopute "eigenvectors" of matrix $$\Sigma$$:`[U, S, V] = svd(Sigma)`, singular value decomposition to find the eigenvalues, more computationally stable, sigma is a n by n matrix(n is original data dimension), U is a n by n, what we want;
     - take first k column from U to from a new matrix $$U_{reduce}$$:
     $$z^i = U_{reduce}^Tx^i $$
     ![](http://i.imgur.com/DWwRfhw.png)
8. Reconstruction from Compressed Representation
    - $$x_{approx} = U_{reduce}z$$
9. Choose proper k of principal components:
    - Average squared projection error: $$\frac{1}{m}\sum\limits_{i=1}^n||x^i - x^i_{approx}||^2$$
    - Total variation in the data: $$\frac{\frac{1}{m}\sum_{i=1}^m ||x^i - x^i_{approx}||^2}{\frac{1}{m}\sum_{i=1}^m ||x^i ||^2} \leq 0.01$$ called: 99%(95%/90%) of variance is retained;
    - ALgorithm: try pca with k = 1, compute the percentage of variance(<= 0.01); repeat and increase k, until you get the 99% variance is retained.
    - `[U, S, V] = svd(Sigma)`, the S matrix, for given k, the percentage of variance can be computed by $$1-\frac{\sum^k_{i=1} S_{ii}}{\sum^n_{i=1} S_{ii}} \leq 0.01$$
    - why we need to retain the variance??
10. Advice for applying PCA
    - supervised learning speedup:
    $$(x^1, y^1), (x^2, y^2)...(x^m, y^m)$$, extract dataset: $$x^1, x^2...x^m\in\mathbb{R}^{10000}$$ to $$z^1, z^2...z^m\in\mathbb{R}^{1000}$$; new dataset $$(z^1, y^1)...$$
    - Applications of PCA:
       - Compression: 1. reduce memory/disk needed to store data; 2. speed up learning algo;
       - Visualization:
       - To prevent overfitting(bad), you should use regularization!
       - Bad Practice:
       ![](http://i.imgur.com/yyKSPjW.png)
       we should do the whole thing without using PCA, try raw daat with whatever you want to , only if theat doesnt do what you want, then implement PCA and consider using $$z^i$$