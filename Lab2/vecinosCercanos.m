function f = vecinosCercanos(Xval,Xent,Yent,k,tipo)

    %%% El parametro 'tipo' es el tipo de modelo de que se va a entrenar

    N=size(Xent,1);
    M=size(Xval,1);
    
    f=zeros(M,1);
    dis=zeros(N,1);
    
    if strcmp(tipo,'class')
        
        for j=1:M
            kaes=[];
            for i=1:N
                dis(i)=distancia(Xval(j,:),Xent(i,:));
            end
            [dis, I]=sort(dis);
            for m=1:k
                kaes=[kaes,(Yent(I(m)))];
            end
            f(j)=mode(kaes);
        end
   
        
    elseif strcmp(tipo,'regress')
        
        for j=1:M
            for i=1:N
                dis(i)=distancia(Xval(j,:),Xent(i,:));
            end
            [dis, I]=sort(dis);
            sum=0;
            for m=1:k
                sum=sum+(Yent(I(m)));
            end
            f(j)=(sum/k);
        end
    end
    
end