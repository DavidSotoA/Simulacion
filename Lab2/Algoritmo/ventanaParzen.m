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
        
        ind1=Yent==1;
        ind2=Yent==2;
        ind3=Yent==3;
        
        X1=Xent(ind1,:);
        X2=Xent(ind2,:);
        X3=Xent(ind3,:);
        
        N1=size(X1,1);
        N2=size(X2,1);
        N3=size(X3,1);
        
        fdp1=[];
        fdp2=[];
        fdp3=[];
        for j=1:M
            sum=0;
            for i=1:N1
                dist=(distancia(Xval(j,:),X1(i,:)))/h;
                k=gaussianKernel(dist);
                sum=sum+k;
            end
            fdp1=[fdp1;(sum/N1)];
        end
        
        for j=1:M 
            sum=0;
            for i=1:N2
                dist=(distancia(Xval(j,:),X2(i,:)))/h;
                k=gaussianKernel(dist);
                sum=sum+k;
            end
            fdp2=[fdp2;(sum/N2)];
        end
        
        for j=1:M 
            sum=0;
            for i=1:N3
                dist=(distancia(Xval(j,:),X3(i,:)))/h;
                k=gaussianKernel(dist);
                sum=sum+k;
            end
            fdp3=[fdp3;(sum/N3)];
        end

        for i=1:M
            if(fdp1(i)>fdp2(i) && fdp1(i)>fdp3(i))
                f(i)=1;
            elseif(fdp2(i)>fdp1(i) && fdp2(i)>fdp3(i))
                f(i)=2;
            else
                 f(i)=3;
            end
            
        end
        
    end
end
