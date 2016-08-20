function W = regresionMultiple(X,Y,eta)

    [N,D]=size(X);
    W=zeros(D,1);
    
   
    for iter = 1:100
        %%fprintf('iter: %d.\n',iter);
        for j=1:D
            sum=0;
            %%fprintf('j: %d.\n',j);
            Wj=W(j);
            for i=1:N
                Xi=X(i,:);
                Fi=Xi*W;
                Yi=Y(i);      
                sum=sum+Xi(j).*(Fi-Yi);
            end
            d=sum/N;
            %%fprintf('d: %f.\n',d);
            Wj=Wj-(eta*d);
            W(j)=Wj;
        end
    end
end