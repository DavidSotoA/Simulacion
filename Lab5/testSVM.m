function Ytest = testSVM(Modelo,Xtest)
    Ytest = simlssvm(Modelo, Xtest);
end
