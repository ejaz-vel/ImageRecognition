function [ Accuracy ] = NNTest()

    % Extract Features From the Images
    [Features] = LoadImages();
    [xTrain, projection] = BestFeats(Features);
    
    % Load The Image Classes
    [yTrain] = LoadLabels();
    
    % Tweak these parameters to get optimum accuracy
    thresholdDifference = 0.0003;
    regularizationRate = 7.5;
    learningRate = 1.2;
    hiddenNodes = 175;
    
    % Train the Neural Net.
    [ weights1, weights2, loss ] = NNTrain(xTrain(1:5000,:), yTrain(1:5000), ...
                                    hiddenNodes, learningRate, ...
                                    regularizationRate, thresholdDifference);
    Model = struct('weights1', weights1, 'weights2', weights2, ...
             'projection', projection);
    save('Model.mat', 'Model');
      
    load('../CIFAR10/data_batch_5.mat');
    predictedClass = classify( Model, data );
    Accuracy = (length(find(predictedClass == labels)) / 10000) * 100;
end