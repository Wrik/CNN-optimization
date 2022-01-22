clear 
clc
close all
%%
current_case = getDefaultCase();
train_case = 'CL'; % it is suggested to train each network separately
if (strcmp(train_case,'CD'))
    trainCNN_CL(current_case.layers);
else
    trainCNN_CD(current_case.layers);  
end
%%
function default_case = getDefaultCase
    load Geometry_Set1.mat   geo_set
    
    [nx, ny, nz, nc, ~] = size(geo_set);
    
    default_case.layers = [ ...
    image3dInputLayer([nx ny nz nc])    % Input layer
    
    convolution3dLayer(4, 5)    % Convolution layer
    batchNormalizationLayer     % Normalization layer
    leakyReluLayer              % Activation layer
    
    convolution3dLayer(3, 10)
    batchNormalizationLayer
    leakyReluLayer

    convolution3dLayer(3, 15)
    batchNormalizationLayer
    leakyReluLayer
    
    convolution3dLayer(3, 20)
    batchNormalizationLayer
    leakyReluLayer
   
    fullyConnectedLayer(1)  % Fully-connected layer
    regressionLayer];   % Output layer

end