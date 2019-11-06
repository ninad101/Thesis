a = load('data_batch_1.mat');
b = load('data_batch_2.mat');
c = load('data_batch_3.mat');
d = load('data_batch_4.mat');
e = load('data_batch_5.mat');


test = load('test_batch.mat');


%% concancate the training data
f.data = [a.data;b.data;c.data;d.data;e.data];
f.labels = [a.labels; b.labels; c.labels;d.labels;e.labels];
%% rbg2gray conversion by taking weighted approach
x= f.data(:,1:1024);
y= f.data(:,1025:2048);
z = f.data(:,2049:3072);

g.data = (0.3 * x) + (0.59 * y) + (0.11 * z);
g.labels = f.labels;


%% for test data
a= test.data(:,1:1024);
b= test.data(:,1025:2048);
c = test.data(:,2049:3072);

%% take mean for rgb2gray by weighted approach
n.data = ( (0.3 * a) + (0.59 * b) + (0.11 * c) );
n.labels = test.labels;

%% save in file
save('CIFAR_train','-struct','g');
save('CIFAR_test','-struct','n'); 
