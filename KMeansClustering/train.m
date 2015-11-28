function [Model] = train()
    [X] = LoadImages();
    [Y] = LoadLabels();
    [M,P] = GetMP(X,Y);
    
    field1 = 'M';
    field2 = 'P';
    Model = struct(field1,M,field2,P);
    save('Model.mat','Model');
end