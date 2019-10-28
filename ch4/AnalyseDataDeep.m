%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Project: WRESTORE
%%
%% Name: Andrew Hoblitzell
%%
%% Date: 2014-10-09
%%
%% Description: This file is used for running our derrp learning
%%              process
%%
%% Revision History
%%
%% 2015-01-26 AH: Created
%%
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [rankData] = AnalyseDataDeep(dataDeep,X)
nout = max(nargout,1) -1;
errorsSetDeep = zeros(7,1); 
pos = length(X(1,:));
len = length(X(:,1));
targetNN = zeros(len,3);
target = zeros(len,1);
traindata = zeros(len,pos-1);

for i=1:len
    posRank = X(i,pos);
    if(posRank==0)
        posRank=1;
    end
    targetNN(i,posRank)=1;
    target(i) = posRank;
end
for i=1:pos-1
    traindata(:,i) = X(:,i);
end

inputData = transpose(traindata); %col Major
targetData = transpose(target);
testData = inputData;
testDataRnk = targetData;
targetNN = transpose(targetNN);
dataDeep = transpose(dataDeep);

SOMNets = cell(7,1);
DeepNets = cell(7,1);
SOMNodes = zeros(7,1);
SOMNodes(1,1) = 10;
SOMNodes(2,1) = 25;
SOMNodes(3,1) = 50;
SOMNodes(4,1) = 100;
SOMNodes(5,1) = 200;
SOMNodes(6,1) = 250;
SOMNodes(7,1) = 500;
trialNos=10;

for i=1:7
    [SOMNets{i}] = createSOM(dataDeep, SOMNodes(i));
    varargout{i+7} = SOMNets{i};
    [rankData{i}] = createDeepNeuralNetClassifier(inputData,targetNN,testData,testDataRnk,SOMNets{i},trialNos);
    %disp(str);
end

for i=1:7
    varargout{i} = rankData{i};
end
