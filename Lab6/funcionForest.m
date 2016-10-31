function Error=funcionForest(Xtrain,Ytrain,Xtest,Ytest)
    NumArboles=10;
    Modelo = TreeBagger(NumArboles,Xtrain,Ytrain);
    Yest = predict(Modelo,Xtest);
    Yest = str2double(Yest);
    Error = 1- sum(Ytest == Yest)/length(Ytest);
end