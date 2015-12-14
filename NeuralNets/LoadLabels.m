%Function used to load labels and concatenate them

function [Labels] = LoadLabels()
Labels = [];
for j = 1:5
    num = num2str(j);
    str = strcat('../CIFAR10/small_data_batch_',num,'.mat');
    load(str);
    Labels = vertcat(Labels,labels);
end

for j = 1:4
    num = num2str(j);
    str = strcat('../CIFAR10/data_batch_',num,'.mat');
    load(str);
    Labels = vertcat(Labels,labels);
end

end