%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Project: WRESTORE
%% 
%% Name: Andrew Hoblitzell
%% 
%% Date: 2014-10-09
%% 
%% Description: This file is used for creating the NN classifier
%% 
%% Revision History
%% 
%% 2014-10-09 AH: Created
%% 
%% 
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [classes,bestnet,errorAbsNN,errorsSet,mserror] = createNeuralNetClassifier(trainData,testData, hiddenLayerSize, numEpochs, maxFails)
numTrainingDimensions=length(trainData(:,1))-1;
trainingDataPoints=trainData(2:numTrainingDimensions,:);
trainingLabels=trainData(1,:);
testDataPoints=testData(2:numTrainingDimensions,:);
testLabels=testData(1,:);

trainingLabelPoints=length(trainingLabels);
numberOfTrainingLabels=max(trainingLabels);
trainingLabelMatrix=zeros(numberOfTrainingLabels,trainingLabelPoints);
testLabelPoints=length(testLabels);
numberOfTestLabels=numberOfTrainingLabels;
testLabelMatrix=zeros(numberOfTestLabels,testLabelPoints);

for(i=1:trainingLabelPoints)
	trainingLabelMatrix(trainingLabels(i),i)=1;
end
for(i=1:testLabelPoints)
        testLabelMatrix(testLabels(i),i)=1;
end

net=patternnet();
net = patternnet(hiddenLayerSize);
net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainParam.epochs=numEpochs;
net.trainParam.show=5;
net.trainParam.min_grad=1e-8;
net.trainParam.max_fail=maxFails;
net.trainParam.sigma=5.0e-7;
net.trainParam.lambda=5.0e-9;
net.performFcn = 'mse';  % Redundant, MSE is default
net.performParam.regularization = 0.01;
net = train(net,trainingDataPoints,trainingLabelMatrix);

%view(net)
outputdata = net(testDataPoints);
perf = mse(net,testLabelMatrix,outputdata);
mserror=perf;
classes = vec2ind(outputdata);
bestnet=net;
errorsSet=perf;
errorAbsNN = perf;

