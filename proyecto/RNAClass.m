function net = RNAClass(X,Y,NumeroNeuronas)
    x = X';
    t = Y';

    % Create a Pattern Recognition Network
    hiddenLayerSize = NumeroNeuronas;
    net = patternnet(hiddenLayerSize);


    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % Train the Network
    [net,tr] = train(net,x,t);





end
