function [ Accuracy ] = LRTest()

    % Extract Features From the Images
    [xTrain] = BestFeats();
    % Load The Image Classes
    [yTrain] = LoadLabels();
    
    % Train the classifier using one vs all methodology
    [weights, loss] = LRTrain(xTrain(1:4500,:), yTrain(1:4500));
    
    % Classify the images
    predictedClass = LRClassify(weights, xTrain(4501:5000,:));
    
    % Calculate Accuracy
    actualClass = yTrain(4501:5000);
    Accuracy = (length(find(predictedClass == actualClass)) / 500) * 100;
end

