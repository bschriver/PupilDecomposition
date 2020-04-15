%%  EXAMPLE RUN TO FIT DATA SET AND EXAMPLE OF RELEVANT PLOTS
% Coded by Brian Schriver, updated 4/14/2020

%% LOAD DATA

load('TestDataSet.mat');
% x : Time-locked regressors initialized as 1s and 0s (simulated)
% y : Observed pupil responses (simulated)

%% DEFINE MODEL PARAMETERS

num_BootStraps = 10;
num_CV_folds = 10;
epoch_check = 1;
epoch_trigger = 5;
alpha_X = .01;
alpha_K = .05;

%% FIT MODEL

DataStore = Decomposition_Fit(x, y, num_BootStraps, num_CV_folds, epoch_check, epoch_trigger, alpha_X, alpha_K);

%% EXAMPLES OF RELEVANT PLOTS

figure;
num_Kernels = size(DataStore.w,3);
for i = 1:num_Kernels
    subplot(2,num_Kernels,i);
    plot(DataStore.w(1,:,i),'k');
    str = sprintf('Kernel #:%d',i);
    title(str);
end
subplot(2,num_Kernels,num_Kernels+1:num_Kernels*2);
hold on;
plot(reshape(y',1,[]),'k')
y_fit = Convolution_Layer(DataStore.x,DataStore.w);
plot(reshape(y_fit',1,[]),'r');
legend('Observed','Fit');
title('Observed and Fit Pupil Traces');

%% Convolution_Layer

function o = Convolution_Layer(x,w)

Num_Filters = size(w,3);
o = zeros(size(x,1),size(x,2));
for i = 1:size(x,1)
    integration = 0;
    for j = 1:Num_Filters
        int_temp = conv(x(i,:,j),w(1,:,j));
        integration = integration + int_temp;
    end
    o(i,:) = integration(1,1:size(x,2));
end

end
