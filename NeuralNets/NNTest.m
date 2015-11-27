function [ Accuracy ] = NNTest()

    % Extract Features From the Images
    [xTrain] = BestFeats();
    % Load The Image Classes
    [yTrain] = LoadLabels();
    
    % Tweak these parameters to get optimum accuracy
    regularizationRate = 0.4;
    learningRate = 1;
    hiddenNodes = 25;
    
    % Train the Neural Net.
    [ weights1, weights2, loss ] = NNTrain(xTrain(1:4500,:), yTrain(1:4500), hiddenNodes, learningRate, regularizationRate);
    
    % Classify the images
    [ predictedClass ] = NNClassify(weights1, weights2, xTrain(4501:5000,:));
    
    % Calculate Accuracy
    actualClass = yTrain(4501:5000);
    Accuracy = (length(find(predictedClass == actualClass)) / 500) * 100;
end