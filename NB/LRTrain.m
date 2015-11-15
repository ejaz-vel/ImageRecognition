function [weights, loss] = LRTrain(xTrain, yTrain)
	[dataSize, featureSize] = size(xTrain);
	labelSize = length(unique(yTrain));
	
    xTrain = [ones(dataSize, 1) xTrain];
	weights = zeros(labelSize,featureSize + 1);
    loss = zeros(labelSize,1);
    learningRate = 0.2;
    regularizationRate = 0;
    nIter = 50;
	
    for label = 1 : labelSize
        output = (yTrain == (label-1));
        for n = 1 : nIter,
            gradient = zeros(featureSize + 1,1);
            loss(label,1) = 0;
            for i = 1 : dataSize,
                hypothesis = xTrain(i,:) * weights(label,:)';
                prediction = sigmoid(hypothesis);
                y = output(i);
                regularizationTerm = regularizationRate * (sum(gradient .^ 2));
                loss(label,1) = loss(label,1) - ((y * log(prediction)) + ((1 - y) * log(1-prediction)));
                loss(label,1) = loss(label,1) + regularizationTerm;
                linearError = prediction - y;
                for j = 1 : (featureSize + 1),
                    gradient(j) = gradient(j) + ((xTrain(i,j) .* linearError) + (regularizationRate .* gradient(j)));
                end;
            end;
            loss(label,1) = loss(label,1) / dataSize;
            gradient = (gradient ./ dataSize);
            weights(label,:) = weights(label,:) - (learningRate .* gradient');	
        end;
    end;
end

