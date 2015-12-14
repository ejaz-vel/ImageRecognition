function [ Accuracy ] = NNTest()

    % Extract Features From the Images
    [Features] = LoadImages();
    [xTrain, projection] = BestFeats(Features);
    
    % Load The Image Classes
    [yTrain] = LoadLabels();
    
    % Tweak these parameters to get optimum accuracy
    thresholdDifference = 0.0002;
    regularizationRate = 0.3;
    learningRate = 1.3;
    hiddenNodes = 200;
    
    load('../CIFAR10/data_batch_5.mat');
    xTest = getImages(data);
    
    % Train the Neural Net.
    [ weights1, weights2, loss ] = NNTrain(xTrain(1:35000,:), yTrain(1:35000), ...
                                    hiddenNodes, learningRate, ...
                                    regularizationRate, thresholdDifference);
    Model = struct('weights1', weights1, 'weights2', weights2, ...
             'projection', projection);
    save('Model.mat', 'Model');
        
    predictedClass = classify( Model, xTest );
    Accuracy = (length(find(predictedClass == labels)) / 10000) * 100;
end

function [Features] =  getImages(data)
    Features = [];
    for i = 1:size(data,1)
        image = reshape(data(i,:),[32,32,3]);
        image = imresize(image,4);
        feat = extract_feature(image);
        Features = horzcat(Features,feat);
    end
    Features = Features';
end