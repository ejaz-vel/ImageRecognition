function [ Accuracy ] = NNTest()

    % Extract Features From the Images
    [Features] = LoadImages();
    [xTrain, projection] = BestFeats(Features);
    
    % Load The Image Classes
    [yTrain] = LoadLabels();
    
    % Tweak these parameters to get optimum accuracy
    thresholdDifference = 0.001;
    regularizationRate = 0.5;
    learningRate = 1;
    hiddenNodes = 50;
    
    % Train the Neural Net.
    [ weights1, weights2, loss ] = NNTrain(xTrain(1:5000,:), yTrain(1:5000), ...
                                     hiddenNodes, learningRate, ...
                                     regularizationRate, thresholdDifference);
    
    Model = struct('weights1', weights1, 'weights2', weights2, ...
                'projection', projection);
    save('Model.mat', 'Model');
    
    % Classify the images
    [ predictedClass ] = NNClassify(weights1, weights2, xTrain(4501:5000,:));
    
    % Calculate Accuracy
    actualClass = yTrain(4501:5000);
    Accuracy = (length(find(predictedClass == actualClass)) / 500) * 100;
end