function [ Feats, projection ] = BestFeats(Features)
    [dataSize, featureSize] = size(Features);
    
    % Perform Mean Normalization on the Feature Matrix.
    % Each pixel can have intensity values from 0 - 255
    for i = 1 : featureSize
        Features(:,i) = Features(:,i) - mean(Features(:,i));
    end
    
    % Get the Covariance Matrix.
    % We will know how each pixel is correlated with other pixels.
    sigma = (1/dataSize) .* (Features' * Features);
    
    % Find The Singular Value Decomposition of the covariance Matrix
    [U, S, V] = svd(sigma);
    
    % Find Some of All Eigen Values
    numOfEigenValues = length(S);
    sumOfEigen = zeros(numOfEigenValues,1);
    for i = 1 : numOfEigenValues
        if i == 1
            sumOfEigen(i) = S(i,i);
        else
            sumOfEigen(i) = sumOfEigen(i-1) + S(i,i);
        end
    end
    
    % Find the minimum dimensions to retain 95% Variance
    dimension = 100;
    for i = 100 : numOfEigenValues
        sum = sumOfEigen(i);
        varianceRetained = sum / sumOfEigen(numOfEigenValues);
        if varianceRetained > 0.99
            dimension = i;
            break;
        end
    end
    
    % Project the Features on the new dimension
    projection = U(:,1:dimension)';
    Feats = projection * Features';
    Feats = Feats';
end