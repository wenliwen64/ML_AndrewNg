# Recommend System
1. Problem Formulation:
    - Examples: Predicting movie ratings: $$n_u$$ = no. of users, $$n_m$$= no. of movies, $$r(i, j) = 1$$ if user j rated movie i. $$y(i,j)$$=rating given by user j to movie i(only if $$r_{ij} = 1$$) ;predict unrated score for each movie by each user.
    - Content based recommender systems: action or romance(x_1- romance intense, x_2-action intense)$$x^i\in\mathbb{R}^3$$, so for each user j, learn a parameter $$\theta^j\in\mathbb{R}^3$$, predict user j as rating movie i with $$(\theta^j)^Tx^i$$ stars for cern movie
    - Formulation: 
    
    $$r(i,j)$$ = 1 if user j has rated movie i(0 otherwise)
    
    $$y^{(i,j)} = $$ rating by user j on movie i
    
    $$\theta^{j}$$ = parameter vector for user j
    
    $$x^i$$ = feature vector for movie i
    
    for user j, movie i, predicted rating by $$(\theta^j)^Tx^i$$
    
    $$m^j$$ = no. of movies rated by user j
    
    to learn  $$\theta^j$$
   
    $$\min\limits_{\theta^j}\frac{1}{2}\sum\limits_{i:r(i,j)=1}((\theta^j)^Tx^i - y^{(i,j)})^2+\frac{\lambda}{2}\sum\limits_{k=1}^n(\theta_k^j)^2$$
    
    repeat these processes, to learn $$\theta^1, \theta^2, \theta^3...\theta^{n_u}$$:
    
    $$\min\limits_{\theta^1, \theta^2,....\theta^{(n_u)}}\frac{1}{2}\sum\limits_{j=1}^{n_u}\sum\limits_{i:r(i,j)=1}((\theta^j)^Tx^i - y^{(i,j)})^2+\frac{\lambda}{2}\sum\limits_{j=1}^{n_u}\sum\limits_{k=1}^n(\theta_k^j)^2$$
    
    Gradient Descents update:
    
    $$\theta^j_k = \theta^j_k - \alpha\sum_{i:r(i,j)=1}((\theta^j)^Tx^i-y^{(i,j)})x^i_k)$$ for k = 0
    
    $$\theta^j_k = \theta^j_k - \alpha[\sum_{i:r(i,j)=1}((\theta^j)^Tx^i-y^{(i,j)})x^i_k) +\lambda\theta^j_k]$$ for k = 0
2. Collaborative Filtering:  
    - Probelm Motivation: suppose we dont know the x1 and x2 values for each movie
   - Optimization Algorithm:
      - Given $$\theta^1, \theta^2....\theta^{n_u}$$, to learn x^{i}:
    $$\min\limits_{x^i}\frac{1}{2}\sum\limits_{j:r(i,j)=1}((\theta^j)^Tx^i - y^{(i,j)})^2+\frac{\lambda}{2}\sum\limits_{k=1}^n(x_k^i)^2$$
      
      - Given $$\theta^1, \theta^2....\theta^{n_u}$$, to learn $$x^1, x^2, x^3...$$:
    $$\min\limits_{x^1, x^2,...x^{(n_m)}}\frac{1}{2}\sum\limits_{i=1}^{n_m}\sum\limits_{j:r(i,j)=1}((\theta^j)^Tx^i - y^{(i,j)})^2+\frac{\lambda}{2}\sum\limits_{i=1}^{n_m}\sum\limits_{k=1}^n(x_k^i)^2$$; Randomly guess $$\theta \rightarrow x \rightarrow \theta \rightarrow x$$
3. Collaborative Filtering Algorithm:
    - Collaborative filtering optimization objective: solve $$x^i$$ and $$\theta^j$$ simultaneously=>minimizing $$x^1, x^2,...x^{n_m}$$ and $$\theta^1, \theta^2,...$$ simultaneously, $$x\in\mathbb{R}^n$$not n+1, let it learn by itself:

    ![](http://i.imgur.com/KrpeyaM.png)
    
    - Initialize to small random values(x and thetas)
    - Minimize J(x, theta)
4. Vectorization: Low randk matrix factorization
    - Y matrix(movie no. by user no.): $$X\Theta^T$$, X = no. movie by no. features, $$\Theta^T$$ = no. user by no. features
    - Finding related movies, for each product i, we learn a feature vector $$x^i\in\mathbb{R}^n$$
    - how to find movies j related to movie i?
5. Implementation details 
    - Users who have not rated any movies
    - Mean Normalization: $$(\theta^j)^Tx^i + \mu_i$$ to guess user5(who doesnot rate anyone)