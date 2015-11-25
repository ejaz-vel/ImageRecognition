function [weights, loss] = LRTrain(xTrain, yTrain)
	[dataSize, featureSize] = size(xTrain);
	labelSize = length(unique(yTrain));
	
    xTrain = [ones(dataSize, 1) xTrain];
	weights = zeros(labelSize,featureSize + 1);
    loss = zeros(labelSize,1);
    regularizationRate = 0;
    nIter = 100;
	
    for label = 1 : labelSize
        learningRate = 10;
        output = (yTrain == (label-1));
        for n = 1 : nIter
            if learningRate > 0.5
                learningRate = learningRate / (n^2);
            end
            gradient = zeros(featureSize + 1,1);
            loss(label,1) = 0;
            for i = 1 : dataSize,
                hypothesis = xTrain(i,:) * weights(label,:)';
                prediction = sigmoid(hypothesis);
                y = output(i);
                regularizationTerm = regularizationRate * ((sum(gradient .^ 2)) / dataSize);
                loss(label,1) = loss(label,1) - ((y * log(prediction)) + ((1 - y) * log(1-prediction))) / dataSize;
                loss(label,1) = loss(label,1) + regularizationTerm;
                linearError = prediction - y;
                gradient = gradient + ((xTrain(i,:)' .* linearError) + (regularizationRate .* gradient)) ./ dataSize;
            end;
            weights(label,:) = weights(label,:) - (learningRate .* gradient');	
        end;
    end;
end

