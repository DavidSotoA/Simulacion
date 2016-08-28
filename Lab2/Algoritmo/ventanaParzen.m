function f = ventanaParzen(Xval,Xent,Yent,h,tipo)

    M=size(Xval,1);
    N=size(Xent,1);
    f=zeros(M,1);

    if strcmp(tipo,'regress')
        
        for i=1:M
            sumN=0;
            sumD=0;
            for j=1:N
                dist=(distancia(Xval(i,:),Xent(j,:)))/h;
                k=gaussianKernel(dist);
                sumN=sumN+(k*Yent(j));
                sumD=sumD+k;
            end
            y=(sumN/sumD);
            f(i)=y;
        end
    
    elseif strcmp(tipo,'class')
        
        X1=[];
        X2=[];
       
        for i=1:N
           if(Yent(i)==1)
               X1=[X1;Xent(i,:)];
           else
               X2=[X2;Xent(i,:)];
           end
        end
        
        N1=size(X1,1);
        N2=size(X2,1);
        fdp1=[];
        fdp2=[];
        for j=1:M
            sum=0;
            for i=1:N1
                dist=distancia(Xval(j,:),X1(i,:));
                k=gaussianKernel(dist/h);
                sum=sum+k;
            end
            fdp1=[fdp1;(sum/N1)];
        end
        
        for j=1:M 
            sum=0;
            for i=1:N2
                dist=distancia(Xval(j,:),X2(i,:));
                k=gaussianKernel(dist/h);
                sum=sum+k;
            end
            fdp2=[fdp2;(sum/N2)];
        end

        for i=1:M
            if(fdp1(i)>fdp2(i))
                f(i)=1;
            else
                f(i)=0;
            end
        end
        
    end
end
