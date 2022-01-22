function results = trainCNN_CL(layers)
% Trains the convolutional neural network given level-set shape inputs
% that have previously been generated. Computes CNN training accuracy with
% coefficient of determination
% Input files: Geometry_Set.mat
%              LLT_Data.mat 
% Output files: wingCNN_CL.mat
% 
% Neil Farvolden and Wrik Mallik
% -------------------------------------------------------------------------
%% Gather the training data
% Geometry input
load Geometry_Set1.mat   geo_set
% Uncomment only with full data
% geo_set1 = geo_set;
% load Geometry_Set2.mat  geo_set
% geo_set1 = cat(5,geo_set1,geo_set);
% load Geometry_Set3.mat  geo_set
% geo_set1 = cat(5,geo_set1,geo_set);
% geo_set = geo_set1;

% Aero coefficients from lifting line analysis
load LLT_Data1.mat       CL  CDi
% Uncomment only with full data
% CL1 = CL; CDi1 = CDi;
% load LLT_Data2.mat       CL  CDi
% CL1 = [CL1;CL]; CDi1 = [CDi1;CDi];
% load LLT_Data3.mat       CL  CDi
% CL1 = [CL1;CL]; CDi1 = [CDi1;CDi];
% CL = CL1; CDi = CDi1;

geo_set_tr = geo_set(:,:,:,:,1:end);
CL_train = CL(1:end)';
%% Pre-process training data
% Normalize input geometry function such that avg=0 and variance=1
avg = mean(geo_set_tr, 'all');
variance = var(geo_set_tr, 0, 'all');
train_input = (geo_set_tr-avg)/sqrt(variance);

% Normalize output CL and CD such that avg=0 and variance=1
avg_CL = mean(CL_train); var_CL = var(CL_train);
train_output = [(CL_train-avg_CL)/sqrt(var_CL)];
%% Training options
options = trainingOptions('adam', 'VerboseFrequency', 20, ...
                'InitialLearnRate', 0.00025, ... % Learning rate
                'MiniBatchSize', 750, ...     % Stochastic batch size
                'MaxEpochs', 3000, ...       % Maximum training iterations
                'Plots','none', ...
                'Shuffle','every-epoch', ...
                'ExecutionEnvironment', 'auto', ... 
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropPeriod', 1, ...
                'LearnRateDropFactor', 0.999);                       

% train network 
network = trainNetwork(train_input, train_output', layers, options);
%% Prediction of training data
% Run training inputs through trained NN
pred_output = predict(network, train_input);

% Post-process NN outputs (de-normalize)
CLpred = pred_output(:,1) * sqrt(var_CL) + avg_CL;

% Determine prediction quality
sseCLtrain = sum((CLpred-CL_train').^2);
CLtrain_coeffdet = 1 - sseCLtrain/(size(CL_train',1)*var(CL_train,1,2));
disp(strcat('Training R^2, CL: ',num2str(CLtrain_coeffdet)))
%% calculating network trainable parameters
nlayers = size(network.Layers,1);
nCNNlayers = (nlayers-3)/3;
ntrainableparams = 0;
for i=1:nCNNlayers
    ntrainableparams = ntrainableparams + numel(network.Layers((i-1)*3+2,1).Weights) + ...
        numel(network.Layers((i-1)*3+2,1).Bias);
end
ntrainableparams = ntrainableparams + numel(network.Layers(nlayers-1,1).Weights) + ...
    numel(network.Layers(nlayers-1,1).Bias);
disp(strcat('Trainable params: ',num2str(ntrainableparams)))
%% Save params into a file for retrieval by optimizer
wingCNN_CL.network = network;
wingCNN_CL.avg = avg;
wingCNN_CL.variance = variance;
wingCNN_CL.var_CL = var_CL;
wingCNN_CL.avg_CL = avg_CL;
wingCNN_CL.CLtrain_coeffdet = CLtrain_coeffdet;
wingCNN_CL.ntrainableparams = ntrainableparams;

results = wingCNN_CL;

% Uncomment if you want to save the network
% save('wingCNN_CL.mat', 'wingCNN_CL');
% save('CL_pred.mat', 'CLpred');
end