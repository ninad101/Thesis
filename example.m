%% Load paths
addpath(genpath('.'));

%% Load data
load mnist;

load mnist_10_per_class;
% load cifar_1000_per_class
% y = load ('CIFAR_test');
% x = load ('CIFAR_train');

% train_X = train_x;
% test_X = test_x;
% train_Y = train_y;
% test_Y = test_y;

%%%%%%%%%%%%%%%%%%%%%%% Uncomment this for CIFAR dataset %%%%%%%%%%%%%%%


% train_X = trainX_1000;
% % % train_X = x.data ;
% test_X = y.data ;
% train_Y = trainY_1000;
% test_Y = y.labels;


%%%%% One-hot Encoding %%%%%%
%%%%%%%%%%%%%%%% Block to modify dataset for the DBN %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Uncomment this block if you are not using mnist_uint8%%%%%%%%%%
train_X = trainX_10;
test_X = testX;
train_Y = trainY_10;
test_Y = testY';

[n,m] = size(test_Y); % finds number of rows for Identity matrix length

q = 10;

X = zeros(n,q);


for i = 1:n   % iterates over rows in test_Y
     for j= 1:q %iterates over columns in X 
       for w = X(i,j)  
         for z = test_Y(i,1) %takes the nth value in test_Y
            if j == z   % check if value equals the column number
                X(i,j)= 1;
            elseif z == 0
                X(i,10)= 1;
            else 
                X(i,j)= 0;
             end 
         end
       end
       
    end
end

newtest_Y = X;
    
[n,m] = size(train_Y); % finds number of rows for Identity matrix length

q = 10;

X = zeros(n,q);


for i = 1:n   % iterates over rows in test_Y
     for j= 1:q %iterates over columns in X 
       for w = X(i,j)  
         for z = train_Y(i,1) %takes the nth value in test_Y
            if j == z   % check if value equals the column number
                X(i,j)= 1;
            elseif z == 0
                X(i,10)= 1;
            else    
                X(i,j)= 0;
             end 
         end
       end
       
    end
end

newtrain_Y = X;
    


    
%  Convert data and rescale between 0 and 0.2
% 
% train_x = double(train_x(1:1000,:)) / 255 * 0.2;
% test_x  = double(test_x(1:1000,:))  / 255 * 0.2;
% train_y = double(train_y(1:1000,:)) * 0.2;
% test_y  = double(test_y(1:1000,:))  * 0.2;


train_x = double(train_X) / 255 * 0.2;
test_x  = double(test_X)  / 255 * 0.2;
train_y = double(newtrain_Y) * 0.2;
test_y  = double(newtest_Y)  * 0.2;

%% Train network
% Setup
rand('seed', 42);
clear edbn opts;
edbn.sizes = [784 400 10];
opts.numepochs = 1000;
opts.batchsize = 100;
% opts.distance  = 53.917058213248346; %uncomment for cifar
% % opts.numspikes = 4000;
% opts.timespan = 10;

% % opts.momentum = 0;
% opts.alpha = 0.5;
% opts.v_thr = 0.09;
% opts.tau_m = 0.9;
% opts.t_ref = 0.003;
% % opts.ngibbs = 1;

[edbn, opts] = edbnsetup(edbn, opts);

% Train
fprintf('Beginning training.\n');
edbn = edbntrain(edbn, train_x, opts);


% Use supervised training on the top layer
edbn = edbntoptrain(edbn, train_x, opts, train_y);

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
%run test
er = edbntest (edbn, test_x, test_y);
%%Estimated Accuracy
fprintf('Scored: %2.2f\n', (1-er)*100);

%print weights
% filename = sprintf('1000shot_cifar_%2.2f-%s.mat',(1-er)*100, date());
% edbnclean(edbn);
% save(filename,'edbn');

% %% Show the EDBN in action
% spike_list = live_edbn(edbn, test_x(7, :), opts);
% output_idxs = (spike_list.layers == numel(edbn.sizes));
% 
% figure(2); clf;
% histogram(spike_list.addrs(output_idxs) - 1, 0:edbn.sizes(end));
% xlabel('Digit Guessed');
% ylabel('Histogram Spike Count');
% title('Label Layer Classification Spikes');
%% Export to xml
% edbntoxml(edbn, opts, 'mnist_edbn');