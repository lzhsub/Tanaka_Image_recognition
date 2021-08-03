
% Loading the Source and the Cifar-10 dataset
addpath(genpath('.'));
reqToolboxes = {'Deep Learning Toolbox'};
if( ~checkToolboxes(reqToolboxes) )
 msg = 'It requires:';
 for i=1:numel(reqToolboxes)
  msg = [msg, reqToolboxes{i}, ', ' ];
 end
 msg = [msg, 'Please install these toolboxes.'];
 error(msg);
end

% help
% https://mathworks.com/help/deeplearning/ref/classify.html

[XTrain, YTrain, XTest, YTest] = load_dataset('cifar10');

N=50000;
if( N>0 )
    XTrain = XTrain(:,:,:,1:N);
    YTrain = YTrain(1:N);
end

disp("Loading the Cifar-10 dataset and training-validation-testing split is Done!")
disp("Ready to initial the Liuzihua Net.... ")

% Prepared for the Loading the Designed Network Architecture.
lgraph = layerGraph();
tempLayers = [
    imageInputLayer([32 32 3],"Name","Cifar10_input")
    convolution2dLayer([3 3],16,"Name","init_conv","Padding","same")
    batchNormalizationLayer("Name","init_BN")
    reluLayer("Name","init_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","Res_1_1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1_1")
    convolution2dLayer([3 3],16,"Name","res1_2","Padding","same")
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu1_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","downsample_1","Padding","same")
    batchNormalizationLayer("Name","down_bn_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","res_1_2","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn_1_2_1")
    reluLayer("Name","relu_1_3")
    convolution2dLayer([3 3],16,"Name","res_1_3","Padding","same")
    batchNormalizationLayer("Name","bn1_2_2")
    reluLayer("Name","relu1_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","downsample_2","Padding","same")
    batchNormalizationLayer("Name","down_bn_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","downsample_3_1","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","down_bn_3_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","res_conv3x3_4","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn5_1")
    reluLayer("Name","relu2_1")
    convolution2dLayer([3 3],32,"Name","res_conv3x3_5","Padding","same")
    batchNormalizationLayer("Name","bn6_1")
    reluLayer("Name","relu2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","res_conv3x3_6_1","Padding","same")
    batchNormalizationLayer("Name","bn5_2")
    reluLayer("Name","relu2_3")
    convolution2dLayer([3 3],32,"Name","res_conv3x3_7_1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn6_2")
    reluLayer("Name","relu2_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","downsample_3_2","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","down_bn_3_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","res_conv3x3_6_2_1","Padding","same")
    batchNormalizationLayer("Name","bn5_3_1")
    reluLayer("Name","relu2_5")
    convolution2dLayer([3 3],32,"Name","res_conv3x3_7_2_1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn6_3_1")
    reluLayer("Name","relu2_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","downsample_3_3_1","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","down_bn_3_3_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5_1")
    reluLayer("Name","relu_5_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","downsample_3_3_2","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","down_bn_3_3_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res_conv3x3_6_2_2","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn5_3_2")
    reluLayer("Name","relu3_1")
    convolution2dLayer([3 3],64,"Name","res_conv3x3_7_2_2","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn6_3_2")
    reluLayer("Name","relu3_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5_2")
    reluLayer("Name","relu_5_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","downsample_3_3_3","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","down_bn_3_3_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res_conv3x3_6_2_3","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5_3_3")
    reluLayer("Name","relu3_3")
    convolution2dLayer([3 3],64,"Name","res_conv3x3_7_2_3","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn6_3_3")
    reluLayer("Name","relu3_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5_3_1")
    reluLayer("Name","relu_5_3_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","downsample_3_3_4","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","down_bn_3_3_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res_conv3x3_6_2_4","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5_3_4")
    reluLayer("Name","relu3_5")
    convolution2dLayer([3 3],64,"Name","res_conv3x3_7_2_4","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn6_3_4")
    reluLayer("Name","relu3_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5_3_2")
    reluLayer("Name","relu_5_3_2")
    averagePooling2dLayer([2 2],"Name","avgpool2d","Padding","same","Stride",[2 2])
    fullyConnectedLayer(128,"Name","fc_1")
    reluLayer("Name","relu")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(10,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
lgraph = connectLayers(lgraph,"init_relu","Res_1_1");
lgraph = connectLayers(lgraph,"init_relu","downsample_1");
lgraph = connectLayers(lgraph,"relu1_2","addition_1/in1");
lgraph = connectLayers(lgraph,"down_bn_1","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_1","res_1_2");
lgraph = connectLayers(lgraph,"relu_1","downsample_2");
lgraph = connectLayers(lgraph,"down_bn_2","addition_2/in1");
lgraph = connectLayers(lgraph,"relu1_4","addition_2/in2");
lgraph = connectLayers(lgraph,"relu_2","downsample_3_1");
lgraph = connectLayers(lgraph,"relu_2","res_conv3x3_4");
lgraph = connectLayers(lgraph,"relu2_2","addition_3/in1");
lgraph = connectLayers(lgraph,"down_bn_3_1","addition_3/in2");
lgraph = connectLayers(lgraph,"relu_3","res_conv3x3_6_1");
lgraph = connectLayers(lgraph,"relu_3","downsample_3_2");
lgraph = connectLayers(lgraph,"down_bn_3_2","addition_4/in2");
lgraph = connectLayers(lgraph,"relu2_4","addition_4/in1");
lgraph = connectLayers(lgraph,"relu_4","res_conv3x3_6_2_1");
lgraph = connectLayers(lgraph,"relu_4","downsample_3_3_1");
lgraph = connectLayers(lgraph,"down_bn_3_3_1","addition_5_1/in1");
lgraph = connectLayers(lgraph,"relu2_6","addition_5_1/in2");
lgraph = connectLayers(lgraph,"relu_5_1","downsample_3_3_2");
lgraph = connectLayers(lgraph,"relu_5_1","res_conv3x3_6_2_2");
lgraph = connectLayers(lgraph,"down_bn_3_3_2","addition_5_2/in2");
lgraph = connectLayers(lgraph,"relu3_2","addition_5_2/in1");
lgraph = connectLayers(lgraph,"relu_5_2","downsample_3_3_3");
lgraph = connectLayers(lgraph,"relu_5_2","res_conv3x3_6_2_3");
lgraph = connectLayers(lgraph,"down_bn_3_3_3","addition_5_3_1/in2");
lgraph = connectLayers(lgraph,"relu3_4","addition_5_3_1/in1");
lgraph = connectLayers(lgraph,"relu_5_3_1","downsample_3_3_4");
lgraph = connectLayers(lgraph,"relu_5_3_1","res_conv3x3_6_2_4");
lgraph = connectLayers(lgraph,"down_bn_3_3_4","addition_5_3_2/in2");
lgraph = connectLayers(lgraph,"relu3_6","addition_5_3_2/in1");

options = trainingOptions('adam', ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.6, ...
    'LearnRateDropPeriod',10, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 100, ...
    'ValidationData',{XTest, YTest}, ...
    'ValidationFrequency', 100, ...
    'Plots','training-progress');

net = trainNetwork(XTrain, YTrain,lgraph,options);

YPred = predict(net,XTrain);
acc = mean_accuracy( YTrain, YPred );
ce = mean_cross_entropy( YTrain, YPred );
fprintf( 'Train mean accuracy: %g\n', acc );
fprintf( 'Train mean cross entropy: %g\n\n', ce );

YPred = predict(net,XTest);
acc = mean_accuracy( YTest, YPred );
ce = mean_cross_entropy( YTest, YPred );
fprintf( 'Test mean accuracy: %g\n', acc );
fprintf( 'Test mean cross entropy: %g\n\n', ce );

if( train1000 )
 disp( '********** ********** ********** **********' );
 disp( '* It was trained with just 1000 samples.' );
 disp( '* Please visit train with 1000 project page: <a href="http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/">http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/</a>' );
 disp( '* ' );
 disp( '* If you want to train with full size of training data, please run as follow:' );
 disp( '* >> train1000 = false; sample_cifar10;' );
 disp( '********** ********** ********** **********' );
end




