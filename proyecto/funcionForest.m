function Error=funcionForest(Xtrain,Ytrain,Xtest,Ytest)
    NumArboles=10;
    Modelo = TreeBagger(NumArboles,Xtrain,Ytrain,'Method','classification');
    Yest = predict(Modelo,Xtest);
    Yest = str2double(Yest);
    Error = sum(Ytest ~= Yest)/length(Ytest);
end