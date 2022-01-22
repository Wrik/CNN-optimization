function results = trainCNN_CD(layers)
% Trains the convolutional neural network given level-set shape inputs
% that have previously been generated. Computes CNN training accuracy with
% coefficient of determination
% Input files: Geometry_Set.mat
%              LLT_Data.mat 
% Output files: wingCNN_CD.mat
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
CD_train = CDi(1:end)';
%% Pre-process training data
% Normalize input geometry function such that avg=0 and variance=1
avg = mean(geo_set_tr, 'all');
variance = var(geo_set_tr, 0, 'all');
train_input = (geo_set_tr-avg)/sqrt(variance);

% Normalize output CL and CD such that avg=0 and variance=1
avg_CD = mean(CD_train); var_CD = var(CD_train);
train_output = [(CD_train-avg_CD)/sqrt(var_CD)];
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
CDpred = pred_output(:,1) * sqrt(var_CD) + avg_CD;

% Determine prediction quality
sseCDtrain = sum((CDpred-CD_train').^2);
CDtrain_coeffdet = 1 - sseCDtrain/(size(CD_train',1)*var(CD_train,1,2));
disp(strcat('Training R^2, CDi: ',num2str(CDtrain_coeffdet)))
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
wingCNN_CD.network = network;
wingCNN_CD.avg = avg;
wingCNN_CD.variance = variance;
wingCNN_CD.var_CD = var_CD;
wingCNN_CD.avg_CD = avg_CD;
wingCNN_CD.CDtrain_coeffdet = CDtrain_coeffdet;
wingCNN_CD.CDval_coeffdet = CDval_coeffdet;
wingCNN_CD.ntrainableparams = ntrainableparams;

results = wingCNN_CD;

% Uncomment if you want to save the network
% save('wingCNN_CD.mat', 'wingCNN_CD');
% save('CD_pred.mat', 'CDpred');
end