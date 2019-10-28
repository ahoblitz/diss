%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Project: WRESTORE
%%
%% Name: Andrew Hoblitzell
%%
%% Date: 2014-10-09
%%
%% Description: This file is used for creating the deep learning
%%              classifier
%%
%% Revision History
%%
%% 2015-01-26 AH: Created
%%
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [rankdata] = createDeepNeuralNetClassifier(inputData,targetData,testData,testDataRnk,nnsom,trialNos)
trainPer =75;
valPer=15;
testPer=15;
%convert Neural Net class data
len = length(targetData);
bestnet = 0;
errorsmin = length(inputData(1,:));
errorsSet = zeros(trialNos,1);
x1 = nnsom(inputData);
for k=1:trialNos
    net = patternnet();
    net.trainParam.max_fail = 200;
    net.divideParam.trainRatio = trainPer/100;
    net.divideParam.valRatio = valPer/100;
    net.divideParam.testRatio = testPer/100;
    net.trainParam.showWindow = false;
    [net] = train(net,x1,targetData);
    x2 = nnsom(testData);
    outputdata = net(x2);
    len = length(x1(1,:));
    rankdata = zeros(1,len);
    for i=1:len
        maxx=0;
        for j=1:3       
            if(maxx<outputdata(j,i))
                maxx = outputdata(j,i);
                rankdata(1,i) = j;
            end
        end
    end
    errorRMS=0;
    for i=1:len
        val = testDataRnk(i)-rankdata(i);
        errorRMS = errorRMS+abs(val);
    end
    errorRMS = errorRMS/len;
    errorRMS = errorRMS/2; %%take average
    if(errorRMS<errorsmin) %new min error
        bestnet = net;
        errorsmin = errorRMS;
    end
    errorsSet(k) = errorRMS;
end
