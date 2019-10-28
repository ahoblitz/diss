%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Project: WRESTORE
%%
%% Name: Andrew Hoblitzell
%%
%% Date: 2014-10-09
%%
%% Description: This file is used for creating the data sets which
%%              will be loaded in to the deep learning process
%%
%% Revision History
%%
%% 2015-01-26 AH: Created
%%
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%seperate datasets
function [dataDeep,data1,data2,data3,data4,data5] = createDeepDataSets(data)
X = getDeepNormalizedWithRank(data);
c = length(data(1,:));
data1 = zeros(120,c);
data2 = zeros(120,c);
data3 = zeros(120,c);
data4 = zeros(240,c);
data5 = zeros(360,c);

for i=1:120
    data1(i,:) = X(i,:);
end
for i=121:240
    id = i-120;
    data2(id,:) = X(i,:);   
end
for i=241:360
    id=i-240;
    data3(id,:) = X(i,:);
end
data4 = [data1; data2];
data5 = [data4; data3];

dataDeep = zeros(length(X(:,1)),c-1);
for i=1:c-1
    dataDeep(:,i) = X(:,i);
end
