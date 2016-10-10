function Modelo = entrenarSVM(X,Y,tipo,boxConstraint,sigma)
    Modelo = trainlssvm({X,Y,tipo,boxConstraint,sigma,'RBF_kernel'});
end
