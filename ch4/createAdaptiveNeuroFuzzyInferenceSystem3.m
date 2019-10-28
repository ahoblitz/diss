%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Project: WRESTORE
%% 
%% Name: Andrew Hoblitzell
%% 
%% Date: 2014-10-09
%% 
%% Description: This file is used for creating the ANFIS classifier
%% 
%% Revision History
%% 
%% 2014-10-09 AH: Created
%% 
%% 
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [anfisPredictions,anfisRmse] = createAdaptiveNeuroFuzzyInferenceSystem(trainData, testData, epochs, targetError, stepSize)
cols = length(trainData(1,:));
rows = length(trainData(:,1));

temp=trainData(:,1);
trainData(:,1)=trainData(:,cols);
trainData(:,cols)=temp;
minimumTrainClass=min(trainData(:,cols));
maximumTrainClass=max(trainData(:,cols));
trainData(:,cols)=(trainData(:,cols)-minimumTrainClass)/(maximumTrainClass-minimumTrainClass);
temp=testData(:,1);
testData(:,1)=testData(:,cols);
testData(:,cols)=temp;
testData(:,cols)=(testData(:,cols)-minimumTrainClass)/(maximumTrainClass-minimumTrainClass);

mfType = 'gbellmf';
trnOpt=[epochs targetError stepSize 0.7 1.1];
in_fis = genfis1(trainData,4,mfType,mfType);
outfis = anfis(trainData,in_fis,trnOpt);
[anfisPredictions] = evalfis(testData(:,1:(cols-1)),outfis);
anfisPredictions=(anfisPredictions*(maximumTrainClass-minimumTrainClass))+minimumTrainClass;
for i=1:length(anfisPredictions)
	if(anfisPredictions(i)<minimumTrainClass)
		anfisPredictions(i)=minimumTrainClass;
	end
	if(anfisPredictions(i)>maximumTrainClass)
		anfisPredictions(i)=maximumTrainClass;
	end
	anfisPredictions(i)=round(anfisPredictions(i));
end
testData(:,cols)=(testData(:,cols)*(maximumTrainClass-minimumTrainClass))+minimumTrainClass;
testData(:,cols);
anfisPredictions;
anfisRmse=sqrt(sum((testData(:,cols)-anfisPredictions(:)).^2)/numel(testData(:,cols)));
