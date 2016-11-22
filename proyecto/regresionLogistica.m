function W = regresionLogistica(X,Y,eta)
    [N,D]=size(X);
    W = zeros(D,1);
    
    for iter = 1:100
        W=W-eta.*((1/N).*((sigmoide(X*W)'-Y)'*X))';
    end

end