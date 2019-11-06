%% Load paths
addpath(genpath('.'));

%% Load data
% load mnist_uint8;
% load cifar_1000_per_class
y = load ('CIFAR_test');
% x = load ('CIFAR_train');

% train_X = train_x;
% test_X = test_x;
% train_Y = train_y;
% test_Y = test_y;

%%%%%%%%%%%%%%%%%%%%%%% Uncomment this for CIFAR dataset %%%%%%%%%%%%%%%


train_X = trainX_10 ;
% train_X = x.data ;
test_X = y.data;
train_Y = trainY_10;
% train_Y = x.labels;
test_Y = y.labels;



%%%%%%%%%%%%%%%% Block to modify dataset for the DBN %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Uncomment this block if you are not using mnist_uint8%%%%%%%%%%
% train_X = trainX;
% test_X = testX;
% train_Y = trainY';
% test_Y = testY';

[n,m] = size(test_Y); % finds number of rows for Identity matrix length

q = 10;

X = zeros(n,q);


for i = 1:n   % iterates over rows in test_Y
     for j= 1:q %iterates over columns in X 
       for w = X(i,j)  
         for z = test_Y(i,1) %takes the nth value in test_Y
            if j == z   % check if value equals the column number
                X(i,j)= 1;
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
            else 
                X(i,j)= 0;
             end 
         end
       end
       
    end
end

newtrain_Y = X;
    


    
% Convert data and rescale between 0 and 0.2
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
rand('seed', 42);
clear edbn opts;
edbn.sizes = [1024 784 500 10];
opts.numepochs = 50;
opts.alpha = 0.005;
[edbn, opts] = edbnsetup(edbn, opts);

opts.momentum = 0.0; opts.numepochs =  2;
edbn = edbntrain(edbn, train_x, opts);
edbn = edbntoptrain(edbn, train_x, opts, train_y);

opts.momentum = 0.8; opts.numepochs = 60;
edbn = edbntrain(edbn, train_x, opts);

edbn = edbntrain(edbn, train_x, opts);
edbn = edbntoptrain(edbn, train_x, opts, train_y);

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, train_x, train_y);
fprintf('Scored: %2.2f\n', (1-er)*100);
filename = sprintf('good_cifar_%2.2f-%s.mat',(1-er)*100, date());
edbnclean(edbn);
save(filename,'edbn');

% opts.momentum = 0.8;
% opts.numepochs = 80;
% edbn = edbntoptrain(edbn, train_x, opts, train_y);
% 
% % Show results
% figure;
% visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
% 
% er = edbntest (edbn, train_x, train_y);
% fprintf('Scored: %2.2f\n', (1-er)*100);
% filename = sprintf('good_cifar_%2.2f-%s.mat',(1-er)*100, date());
% edbnclean(edbn);
% save(filename,'edbn');

% %% Show the EDBN in action
% spike_list = live_edbn(edbn, test_x(1, :), opts);
% output_idxs = (spike_list.layers == numel(edbn.sizes));
% 
% figure(2); clf;
% hist(spike_list.addrs(output_idxs) - 1, 0:edbn.sizes(end));
% 
% %% Export to xml to load into JSpikeStack
% edbntoxml(edbn, opts, 'mnist_edbn');