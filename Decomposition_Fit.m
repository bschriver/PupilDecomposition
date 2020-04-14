function DataStore = Decomposition_Fit(x, y, num_BootStraps, num_CV_folds, epoch_check, epoch_trigger, alpha_X, alpha_K)

%%  Decomposition_Fit for simulatenously learning kernels and time-locked regressor weights
% Coded by Brian Schriver, updated 4/14/2020

% INPUT VARIABLES
% x : Regressor Matrix (n,t,k); n trials, t time points, k components
    % (associated with kernels)
% y : Observation Matrix (n,t); n trials, t time points
    % num_BootStraps : Number of iterations for bootstrap aggregating
    % (number of naive models to fit, which will be averaged)
% num_CV_folds : Number of folds to break data set into for
    % cross-validationf
% epoch_check : Frequency to check if MSE has increased (ex. 2 means
    % check every other iteration
% epoch_trigger : Number of consecutive increases in MSE (testing data set)
    % required to stop model fitting
% alpha_X : Learning rate for regressor weights
% alpha_K : Learning rate for kernels

% OUTPUT
% DataStore.x : Fit regressor matrix (n,t,k); n trials, t time points, k
    % components (associated with kernels)
% DataStore.w : Fit kernel matrix (1,t,k); t time points, k components
    % (associated with kernels)

% DataStore.Strap(num_BootStraps).CV(num_CV_folds)... : Provides output
    % data broken up for each bagging iteration (.Strap) and for each
    % cross-validation iteration (.CV) within each bagging iteration
% DataStore.Strap(num_BootStraps).CV(num_CV_folds).w : Fit kernel matrix
    % (number of SGD iterations, t, k); Each row corresponds to an
    % iteration of SGD with the final kernels as the last row
% DataStore.Strap(num_BootStraps).CV(num_CV_folds).mse : mean squared error
    % vector (number of SGD iterations); MSE from fitting training data
% DataStore.Strap(num_BootStraps).CV(num_CV_folds).val_error : Test data
    % error vector (number of SGD iterations); Error from test data

%% EXAMPLE RUN PARAMETERS (MUST SUPPLY X and Y)

% num_BootStraps = 10;
% num_CV_folds = 10;
% epoch_check = 1;
% epoch_trigger = 3;
% alpha_X = .01;
% alpha_K = .05;
% DataStore = Decomposition_Fit(x, y, num_BootStraps, num_CV_folds, epoch_check, epoch_trigger, alpha_X, alpha_K);

%%

tic;

% INITIAILIZE STORAGE
x_Fit = zeros(size(x));
w_Fit = zeros(1,size(x,2),size(x,3));

% INIITIALIZE BOOTSTRAP AGGREGATING LOOP
for ss = 1:num_BootStraps
    
    %% SPLIT X INTO TRAINING AND VALIDATION SET
    rand_indx = randperm(size(x,1))';
    
    for sss = 1:num_CV_folds
            
            if sss < num_CV_folds
                JK_indx = ((sss-1) * floor(size(rand_indx,1)/num_CV_folds)) + 1:sss * floor(size(rand_indx,1)/num_CV_folds);
            else
                JK_indx = ((sss-1) * floor(size(rand_indx,1)/num_CV_folds)) + 1:length(rand_indx);
            end
            train_indx = rand_indx;
            train_indx(JK_indx) = [];
            val_indx = rand_indx(JK_indx);
            
            x_train = x(train_indx,:,:);
            y_train = y(train_indx,:,:);
            x_val = x(val_indx,:,:);
            y_val = y(val_indx,:,:);
            
            %% INITIALIZATIONS
            
            % INITIALIZE KERNELS
            w = Initialize_Kernels(x_train);
            
            % INITIALIZE NESTEROV-ACCELERATED ADAPTIVE MOMENT ESTIMATION VARIABLES/STORAGE            
            m_K = zeros(1,size(w,2),size(w,3));
            n_K = zeros(1,size(w,2),size(w,3));
            m_X = zeros(size(x_train));
            n_X = zeros(size(x_train));
            
            % INITIAILZE ERROR VARIABLES/STORAGE
            mse = [];
            mse_temp = Inf;
            val_err = Inf;
            
            x_val_preError1 = [];
            x_train_preError1 = [];
            w_preError1 = [];
            y_val_preError1 = [];
            y_train_preError1 = [];
            
            %% FORWARD CONVOLUTION/BACKPROPAGATION TO UPDATE REGRESSOR WEIGHTS AND KERNELS USING SGD
            iteration = 1;
            t = 1;
            mse_check = 0;
            while mse_check == 0
                [x_train,y_train,w,m_K,n_K,t,mse] = StochasticGradientDescent_NADM(x_train,y_train,w,iteration,alpha_K,m_K,n_K,t,alpha_X,mse);
                if rem(iteration,1) == 0
                    fprintf('\n Completed Iteration %d, MSE = %d', iteration, mse(iteration));
                    if iteration ~= 1
                        fprintf('\n Improvement = %f',mse(iteration-1) - mse(iteration));
                    end
                end
                mse_temp = mse(iteration);
                iteration = iteration + 1;
                
                %% EVALUATE ERROR
                x_train_preError1 = x_train;
                x_val_preError1 = x_val;
                y_train_preError1 = y_train;
                y_val_preError1 = y_val;
                w_preError1 = w;

                x_train_preError2 = x_val_preError1;
                x_val_preError2 = x_train_preError1;
                y_train_preError2 = y_train_preError1;
                y_val_preError2 = y_val_preError1;
                w_preError2 = w_preError1;
                
                [x_train, x_val, y_train, y_val, w, val_err, mse_check]  = Evaluate_Error(x_train, x_val, y_train, y_val, w, x_train_preError1, x_val_preError1, y_train_preError1, y_val_preError1, w_preError1, x_train_preError2, x_val_preError2, y_train_preError2, y_val_preError2, w_preError2, val_err, iteration, epoch_check, epoch_trigger, mse_check);
                fprintf('\n Validation Error = %f', val_err(end));
                fprintf('\n CV Partition = %d/%d', sss, num_CV_folds);
                fprintf('\n Boostrap = %d/%d', ss, num_BootStraps);
                fprintf('\n');
                
            end
            
            % UPDATE BETA WEIGHTS
            for i_r = 1:length(rand_indx)
                if sum(train_indx == i_r)
                    x_Fit(i_r,:,:) = x_Fit(i_r,:,:) + x_train(train_indx == i_r,:,:);
                end
            end
            
            % SAVE DATA
            indx = sum(w(:,:,1),2) == 0;
            w(indx,:,:) = [];
            w_Fit = w_Fit + w(end,:,:);
            
            DataStore.Strap(ss).CV(sss).w = w;
            DataStore.Strap(ss).CV(sss).mse = mse;
            DataStore.Strap(ss).CV(sss).val_err = val_err;
    
    end
end

%% CONSOLIDATE OUTPUT DATA
w_Fit = w_Fit ./ (num_BootStraps * num_CV_folds);
x_Fit = x_Fit ./ (num_BootStraps*(num_CV_folds-1));

DataStore.x = x_Fit;
DataStore.w = w_Fit;

fprintf('\n'); toc;

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize_Kernels

function w = Initialize_Kernels(x)

rand_u = 0;
rand_std = .1;
w = zeros(1,size(x,2),size(x,3));
for i = 1:size(w,3)
    w(:,:,i) = normrnd(rand_u,rand_std,[1,size(x,2)]);
end
w = max(w,0);
num_max_iterations = 1000;
w_temp = zeros(num_max_iterations,size(w,2),size(w,3));
w_temp(1,:,:) = w;
w = w_temp;

end

%% StochasticGradientDescent_NADM

function [x,y,w,m_K,n_K,t,mse] = StochasticGradientDescent_NADM(x,y,w,i,alpha_K,m_K,n_K,t,alpha_X,mse)

% RANDOMIZE ORDER
indx = randperm(size(x,1))';
x = x(indx,:,:);
y = y(indx,:);

% NESTEROV ACCELERATED GRADIENT
mu = .99;
v = .999;
eps = 1e-9;
u_t = @(it) mu .* ( 1 - (.5 .* (.96 .^ (it/250))));

% UPDATE FOR EVERY TRIAL INDIVIDUALLY
for iii = 1:size(x,1)
    % PREDICTION
    output = Convolution_Layer(x(iii,:,:),w(i,:,:));
    
    % LOSS FUNCTION/CALCULATE GRADIENT
    dL_dK = Gradient_dK(x(iii,:,:),y(iii,:),output) ./ size(y,1);
    dL_dX = Gradient_dX(x(iii,:,:),w(i,:,:),y(iii,:),output);
    
    % UPDATE HYPERPARAMETERS
    for ii = 1:t
       if ii == 1
           prod = u_t(ii);
       else
           prod = prod .* u_t(ii);
       end
    end
    g_hat = dL_dK ./ (1 - prod);
    m_K = (mu .* m_K) + ((1 - mu) .* dL_dK);
    for ii = 1:t+1
       if ii == 1
           prod = u_t(ii);
       else
           prod = prod .* u_t(ii);
       end
    end
    m_K_hat = m_K ./ (1 - prod);
    n_K = (v .* n_K) + ((1 - v) .* (dL_dK .^ 2));
    n_K_hat = n_K ./ (1 - (v .^ t));
    m_K_bar = ((1 - u_t(t)) .* g_hat) + (u_t(t+1) * m_K_hat);
    t = t + 1;
    
    % UPDATE KERNELS
    for ii = 1:size(w,3)
        update = alpha_K .* (m_K_bar(:,:,ii) ./ (sqrt(n_K_hat(:,:,ii)) + eps));
        w(i,:,ii) = w(i,:,ii) - update;
    end
    % UPDATE WEIGHTS
    for ii = 1:size(x,3)
        x(iii,:,ii) = x(iii,:,ii) - (alpha_X .* dL_dX(:,:,ii));
    end
end
    
% REORGANIZE
x(indx,:,:) = x;
y(indx,:) = y;
w(i+1,:,:) = w(i,:,:);

% PREDICTION
output = Convolution_Layer(x,w(i,:,:));
mse_temp = sum(sum((y-output).^2)/size(y,2))/size(y,1);
mse = [mse; mse_temp];

end


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

%% Gradient_dK

function dL_dK = Gradient_dK(x,y,output)

dL_dK = zeros(1,size(x,2),size(x,3));
for ii = 1:size(dL_dK,3)
    for jj = 1:size(dL_dK,2)
        J = zeros(size(x,3),size(x,2));
        J(ii,jj) = 1;
        dL_dK_temp = 0;
        for n = 1:size(x,1) % Number of trials
            for t = 1:size(x,2) % length t
                % Calculate dY_dK
                dY_dK = 0;
                for i = 1:size(x,3)
                    for r = 1:t
                    %for r = max(t-(size(x,3)-1),1):t
                        dY_dK_temp = x(n,r,i) * J(i,t+1-r);
                        dY_dK = dY_dK + dY_dK_temp;
                    end
                end
                error = output(n,t) - y(n,t);
                dL_dK_temp = dL_dK_temp + (error * dY_dK);
            end
        end
%         dL_dK(:,jj,ii) = ((2/size(y,1))*dL_dK_temp) ./ size(y,1);
        dL_dK(:,jj,ii) = ((2/size(y,1))*dL_dK_temp);
    end
end

end

%% Gradient_dX

function dL_dX = Gradient_dX(x,w,y,output)

dL_dX = zeros(size(x));
for ii = 1:size(dL_dX,3)
    for jj = 1:size(dL_dX,1)
        for kk = 1:size(dL_dX,2)
            if x(jj,kk,ii) ~= 0
                J = zeros(size(y,1),size(w,2),size(w,3));
                J(jj,kk,ii) = 1;
                dL_dX_temp = 0;
                for n = 1:size(x,1) % Number of trials
                    for t = 1:size(w,2) % length t
                        % Calculate dY_dK
                        dY_dX = 0;
                        for i = 1:size(x,3)
                            for r = 1:t
                            %for r = max(t-(size(x,3)-1),1):t
                                dY_dX_temp = J(n,r,i) * w(1,t+1-r,i);
                                dY_dX = dY_dX + dY_dX_temp;
                            end
                        end
                        error = output(n,t) - y(n,t);
                        dL_dX_temp = dL_dX_temp + (error * dY_dX);
                    end
                end
                if x(jj,kk,ii) ~= 0
                    dL_dX(jj,kk,ii) = (2/size(y,1)) * dL_dX_temp;
                end
            end
        end
        % FORCE BOXCAR
        dL_dX(jj,dL_dX(jj,:,ii) ~= 0,ii) = mean(dL_dX(jj,dL_dX(jj,:,ii) ~= 0,ii));
    end
end

end

%% Evaluate_Error

function [x_train, x_val, y_train, y_val, w, val_err, mse_check]  = Evaluate_Error(x_train, x_val, y_train, y_val, w, x_train_preError1, x_val_preError1, y_train_preError1, y_val_preError1, w_preError1, x_train_preError2, x_val_preError2, y_train_preError2, y_val_preError2, w_preError2, val_err, iteration, epoch_check, epoch_trigger, mse_check)

val_temp = 0;
for i_v = 1:size(x_val,1) % for each trial
    x_val_temp  = zeros(size(x_val,3),size(x_train,2));
    for i_k = 1:size(x_val,3) % for each kernel
        x_val_temp_temp = conv(x_val(i_v,:,i_k),w(iteration,:,i_k));
        x_val_temp(i_k,:) = x_val_temp_temp(1:size(x_train,2));
        b_pred = pinv(x_val_temp') * y_val(i_v,:)';
        val_temp_update = sum(abs(y_val(i_v,:) - (b_pred' * x_val_temp)) .^ 2) ./ size(y_val,2);
        val_temp = val_temp + val_temp_update;
    end    
end

val_temp = val_temp ./ size(x_val,1);
val_err = [val_err; val_temp];
if iteration > (epoch_trigger * epoch_check) && rem(iteration,epoch_check) == 0
    
    mse_check_temp = 0;
    for ii = 1:epoch_trigger
        val_check = val_err(end-((ii-1) * epoch_check)) > val_err(end-(epoch_trigger * epoch_check));
        mse_check_temp = mse_check_temp + val_check;
    end
    mse_check = mse_check_temp == epoch_trigger;
    
    if mse_check == epoch_trigger
        x_val = x_val_preError2;
        x_train = x_train_preError2;
        w = w_preError2;
        y_val = y_val_preError2;
        y_train = y_train_preError2;
    else
        if sum(sum(sum(x_val_preError1))) ~= 0
            x_train_preError2 = x_val_preError1;
            x_val_preError2 = x_train_preError1;
            y_train_preError2 = y_train_preError1;
            y_val_preError2 = y_val_preError1;
            w_preError2 = w_preError1;
        end
        
        x_train_preError1 = x_train;
        x_val_preError1 = x_val;
        y_train_preError1 = y_train;
        y_val_preError1 = y_val;
        w_preError1 = w;
    end
    
end

end
