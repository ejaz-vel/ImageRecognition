function [ Accuracy ] = LRTest()

    % Extract Features From the Images
    [Features] = LoadImages();
    [xTrain, projection] = BestFeats(Features);
    
    % Load The Image Classes
    [yTrain] = LoadLabels();
    
    % Learning Parameters. Tweak these values to get optimum accuracy
    regularizationRate = 0.4;
    initialLearningRate = 10;
    stableLearningRate = 0.5;
    thresholdDifference = 0.0005;
    
    % Train the classifier using one vs all methodology
    [weights, loss] = LRTrain(xTrain(1:5000,:), yTrain(1:5000), regularizationRate, ...
                        initialLearningRate, stableLearningRate, thresholdDifference);
                    
    Model1 = struct('weights', weights, 'projection', projection);
    save('Model1.mat', 'Model1');                
                    
    load('../CIFAR10/data_batch_5.mat');
    predictedClass = classify1( Model1, data );
    
    % Calculate Accuracy
    actualClass = labels;
    Accuracy = (length(find(predictedClass == actualClass)) / 10000) * 100;
end

