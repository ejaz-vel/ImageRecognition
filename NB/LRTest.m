function [ Accuracy ] = LRTest()
    [xTrain] = BestFeats();
    [yTrain] = Load_Labels();
    
    [weights, loss] = LRTrain(xTrain(1:4500,:), yTrain(1:4500));
    predictedClass = LRClassify(weights, xTrain(4501:5000,:));
    actualClass = yTrain(4501:5000);
    
    Accuracy = length(find(predictedClass == actualClass)) / 500;
end

