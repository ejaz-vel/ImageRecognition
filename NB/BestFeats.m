%This is an alternative function to using Compute feats
function [Feats] = BestFeats()
    load('HOG_Features.mat');
    
    F1 = cell2mat(Features(1));
    F2 = cell2mat(Features(2));
    F3 = cell2mat(Features(3));
    F4 = cell2mat(Features(4));
    F5 = cell2mat(Features(5));
    Features = horzcat(F1,F2,F3,F4,F5);
    Features = Features';
    [dataSize, featureSize] = size(Features);
    
    % Perform Mean Normalization on the Feature Matrix.
    % Each pixel can have intensity values from 0 - 255
    for i = 1 : featureSize
        Features(:,i) = (Features(:,i) - mean(Features(:,i))) ./ 255;
    end
    
    % Get the Covariance Matrix.
    % We will know how each pixel is correlated with other pixels.
    sigma = 0.5 .* (Features' * Features);
    
    % Find The Singular Value Decomposition of the covariance Matrix
    [U, S, V] = svd(sigma);
    
    % Find Some of All Eigen Values
    sumOfEigen = 0;
    for i = 1 : length(S)
        sumOfEigen = sumOfEigen + S(i,i);
    end
    
    % Find the minimum dimensions to retain 90% Variance
    dimension = 100;
    for i = 100 : length(S)
        sum = 0;
        for j = 1 : i
            sum = sum + S(j,j);
        end
        varianceRetained = sum / sumOfEigen;
        if varianceRetained > 0.80
            dimension = i;
            break;
        end
    end
    
    % Project the Features on the new dimension
    Feats = Features * U(:,1:dimension);
end