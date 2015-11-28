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
                    
    % Classify the images
    predictedClass = LRClassify(weights, xTrain(4501:5000,:));
    
    % Calculate Accuracy
    actualClass = yTrain(4501:5000);
    Accuracy = (length(find(predictedClass == actualClass)) / 500) * 100;
end

