function W = regresionLogistica(X,Y,eta)

[N,D]=size(X);
W = zeros(D,1);

    for iter = 1:100
%          for j=1:D
%             sum=0;
%             Wj=W(j);
%             for i=1:N
%                 Xi=X(i,:);
%                 Fi=Xi*W;
%                 Yi=Y(i);   
%                 g = 1./(1 + exp(-Fi));
%                 sum=sum+Xi(j).*(g-Yi);
%             end           
%             d=sum/N;
%             Wj=Wj-(eta*d);
%             W(j)=Wj;
%         end
        
       W=W-eta.*((1/N).*((sigmoide(X*W)'-Y)'*X))';
    end

end