function S = sigmoide(X)

        S=[];
    for i=1:length(X)
        xi=X(i);
        S =[S, 1./(1 + exp(-xi))];
    end
    
end