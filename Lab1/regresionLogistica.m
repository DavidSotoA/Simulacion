function W = regresionLogistica(X,Y,eta)

    [N,D]=size(X);
    W = zeros(D,1);
    
    X1=[];
    X2=[];

    for i=1:N
       if(Y(i)==1)
           X1=[X1;X(i,:)];
       else
           X2=[X2;X(i,:)];
       end
    end
    [s1,d1]=size(X1);
    Z1=zeros(s1,1);
    
    [s2,d2]=size(X2);
    Z2=zeros(s2,1);
   
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


    figure(1)
% 

     y=-0.5:0.1:6;
     x = -((y*W(3)+W(2))/W(1));
     plot(X1(:,1).^2,X1(:,2).^2,'*',X2(:,1).^2,X2(:,2).^2,'o',x,y);

    end

end