function [ prediction ] = LRClassify( weights, xTest )
    [dataSize, featureSize] = size(xTest);
    xTest = [ones(dataSize, 1) xTest];
    
    probability = sigmoid(xTest * weights');
    [value, index] =  max(probability, [], 2);
    prediction = index - 1;
end
