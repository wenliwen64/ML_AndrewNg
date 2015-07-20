# Decision Trees(UBC courses on youtube)
1. Decision trees(forest)
    - a general tree structure: root node, internal node(split node), terminal node(leaf node)
    - each decision is a line(each line is only slightly better than random guess, combine many of these would be a better algorithms)
    - how to pick attribute(nodes)
    - Entropy:
    
    $$H(\frac{p}{p+n}, \frac{n}{p+n}) = -\frac{p}{p+n}log_2(\frac{p}{p+n})-\frac{n}{p+n}log_2(\frac{n}{p+n})$$
    
    - for a chosen attribute A, with K distinct values, divides the training set E into subsets E1, E2, ...Ek, Expected Entropy(EH)

    $$EH(A) = \sum_{i=1}^K \frac{p_i+n_i}{p+n}H(\frac{p_i}{p_i+n_i}, \frac{n_i}{p_i+n_i})$$
    
    - Information gain(IG) or reduction entropy for this attribute is:
    
    $$IG(A) = H(\frac{p}{p+n}, \frac{n}{p+n}) - EH(A)$$
    
    - choose the one increase information the most
2. Using reduction in entropy as a criterion for constructing decision trees;
3. Application of decision trees to classification;
