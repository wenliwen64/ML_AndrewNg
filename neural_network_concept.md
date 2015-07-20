#Neural Network

1. Hypothesis Representation / Non-linear hypothesis
   - why complex non-linear hypothesis:

       1. Some label distribution require non-linear;
       1. If the feature number is large, linear regression including non-linear terms would be computationally expensive, that is why we need neural network;
  
2. Neuron model:
   - input, bias unit, input wires, activation function(h_theta), output
  
3. Neural network:
   - input layer: $$x_0(bias unit), x_1, x_2, x_3...$$
   - hidden layer(in between input and output layer): $$a_0^{(2)}, a_1^{(2)}, a_2^{(2)}, a_3^{(2)}...$$
   - output layer: final layer;
   - Mathematical definition how to represent hypothesis in neural network(matrix computation): $$a^{(j+1)} = g(\Theta^{(j)}a^{(j)})=g(z^{(j+1)})$$, input layer is $$a^{(1)}$$ , g is sigmoid function.
   - next layer takes this layers output as features.
4. Use neural network to represent logic unit:
   - by applying different weight between bias unit and input;
   - XOR: exclusive or; XNOR: not(XOR) 

5. Multi-Class Classification:
   - multiclass classification using neural networks is an extension of one vs. all method.

6. MATLAB: fmincg(better for large no. of features)
7. Symmetry Breaking