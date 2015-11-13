%Function used to load labels and concatenate them

function [Labels] = Load_Labels()
Labels = [];
for j = 1:5
    num = num2str(j);
    str = strcat('../subset_CIFAR10/small_data_batch_',num,'.mat');
    load(str);
    Labels = vertcat(Labels,labels);
end

end