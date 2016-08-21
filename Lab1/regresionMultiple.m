function W = regresionMultiple(X,Y,eta,Xtest,Ytest)

    [N,D]=size(X);
    W=zeros(D,1);
    ECM=[];
    ECMtest=[];

    for iter = 1:100
        %%%Algoritmo de aprendizaje%%%%%%%%%%%%
        for j=1:D
            sum=0;
            Wj=W(j);
            for i=1:N
                Xi=X(i,:);
                Fi=Xi*W;
                Yi=Y(i);      
                sum=sum+Xi(j).*(Fi-Yi);
            end           
            d=sum/N;
            Wj=Wj-(eta*d);
            W(j)=Wj;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Calcular error de la iteracion iter con las muestras de
        %entrenamiento
        a=((X*W)-Y).^2;
        Esum=0;
        for i=1:N
            Esum=Esum+a(i);
        end
        ECM=[ECM,(1/(N)*Esum)];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
        
        %Calcular error de la iteracion iter con las muestras de
        %test
        a=((Xtest*W)-Ytest).^2;
        Esum=0;
        for i=1:length(Xtest)
            Esum=Esum+a(i);
        end
        ECMtest=[ECMtest,(1/(length(Xtest))*Esum)];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    subplot(1,2,1);
    plot(1:100,ECM,'Color',[0,0.7,0.9]);
    ylabel('ECM');
    xlabel('Iteraciones');
    title('ECM vs iteraciones para muestras de entrenamiento');
    
    subplot(1,2,2)
    plot(1:100,ECMtest,'Color',[0,0.7,0.9]);
    ylabel('ECM');
    xlabel('Iteraciones');
    title('ECM vs iteraciones para muestras de test');
end