function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
CSet = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300];
sigmaSet = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
noRow = length(CSet);
noCol = length(sigmaSet);
valErr = zeros(noRow, noCol);
for i = 1:noRow%length(CSet)
    for j = 1:noCol%length(sigmaSet)
        model= svmTrain(X, y, CSet(i), @(x1, x2) gaussianKernel(x1, x2, sigmaSet(j)));
        predictions = svmPredict(model, Xval);
        valErr(i, j) = mean(double(predictions ~= yval));
    end
end

[minErr, minIndex] = min(valErr(:));
[min_row, min_col] = ind2sub(size(valErr), minIndex);
fprintf('minErr = %f, minIndex = %d, %d', minErr, min_row, min_col);
C = CSet(min_row);
sigma = sigmaSet(min_col);
valErr

% =========================================================================

end
