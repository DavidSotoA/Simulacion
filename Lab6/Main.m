clc
clear all
close all

punto=input('Seleccione el punto a ejecutar: ');

load('DatosSeleccion.mat');

if (punto == 1)   
	X = zscore(X);
    
    alpha = 0.05;
    [correlacion,p]= corrcoef([X,Y],'alpha',alpha);
    
    indicesClase0 = find(Y == 0);
    indicesClase1 = find(Y == 1);
    
    mediaClase0 = mean(X(indicesClase0,:) ,1);
    mediaClase1 = mean(X(indicesClase1,:) ,1);
    
    media = [mediaClase0; mediaClase1];
    
    varClase0 = var(X(indicesClase0,:) ,1);
    varClase1 = var(X(indicesClase1,:) ,1);
    
    varianza = [varClase0;varClase1];
    
    coef = zeros(1,18);
    
    for i=1:2
        for j=1:2
            if (j ~= i)
                numerador = (media(i,:) - media(j,:)).^2;
                denominador = varianza(i,:) + varianza(j,:);
                coef = coef + (numerador./denominador);
            end
        end
    end
    
    Texto = ['Indice de Fisher: ', num2str(coef)];
    disp(Texto);
    
    coefN = coef./max(coef);
    Texto = ['Indice de Fisher Normalizado: ', num2str(coefN)];
    disp(Texto);
    
elseif (punto == 2)
    
    Rept=10; 
    NumMuestras=size(X,1); 
    EficienciaTest=zeros(1,Rept);
    
    opciones = statset('display','iter');
    sentido = 'forward';
    
%     [caracteristicasElegidas, proceso] = sequentialfs(@funcionForest,X,Y,'direction',sentido,'options',opciones);
%     X = X(:, caracteristicasElegidas);
    
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Se normalizan los datos %%%

        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Estas instrucciones lo que hacen es que realizan la seleccion de caracteristicas en cada iteracion. Pero esto es muy demorado!!
        
% % % % %         [caracteristicasElegidas, proceso] = sequentialfs(@funcionForest,Xtrain,Ytrain,'direction',sentido,'options',opciones);
% % % % %         XReducidas = X(:, caracteristicasElegidas);
% % % % %         [NumMuestras,~] = size(XReducidas);
% % % % %         indices=randperm(NumMuestras);
% % % % %         porcionEntrenamiento = round(NumMuestras*0.7);
% % % % %         Xtrain=XReducidas (indices(1:porcionEntrenamiento),:);
% % % % %         Xtest=XReducidas(indices(porcionEntrenamiento+1:end),:);
% % % % %         Ytrain=Y(indices(1:porcionEntrenamiento),:);
% % % % %         Ytest=Y(indices(porcionEntrenamiento+1:end),:);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        NumArboles=10;
        Modelo = TreeBagger(NumArboles,Xtrain,Ytrain);
        
        Yest = predict(Modelo,Xtest);
        Yest = str2double(Yest);
        
        EficienciaTest(fold) = sum(Ytest == Yest)/length(Ytest);
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    
elseif (punto == 3)
    
    Rept=10;
    NumMuestras=size(X,1);
    EficienciaTest=zeros(1,Rept);
    
    caracteristicasElegidas = SelectionGA(@FitnessSelection,X,Y);
    X = X(:, caracteristicasElegidas);
    
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Se normalizan los datos %%%

        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
        
        NumArboles=10;
        Modelo = TreeBagger(NumArboles,Xtrain,Ytrain);
        
        Yest = predict(Modelo,Xtest);
        Yest = str2double(Yest);
        
        EficienciaTest(fold) = sum(Ytest == Yest)/length(Ytest);
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    
elseif (punto == 4)
    
    Rept=10;
    NumMuestras=size(X,1);
    EficienciaTest=zeros(1,Rept);
    
    umbralPorcentajeDeVarianza = 85;
    
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        indices=particion.training(fold);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold));
        Ytest=Y(particion.test(fold));
        
        %%% Se normalizan los datos %%%

        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [coefCompPrincipales,scores,covarianzaEigenValores,~,porcentajeVarianzaExplicada,~] = pca(Xtrain);
        
        numVariables = length(covarianzaEigenValores);
        numCompAdmitidos = 0;
        
        porcentajeVarianzaAcumulada = zeros(numVariables,1);
        puntosUmbral = ones(numVariables,1)*umbralPorcentajeDeVarianza;
        ejeComponentes = 1:numVariables;
        
        for k=1:numVariables
            
            porcentajeVarianzaAcumulada(k) = sum(porcentajeVarianzaExplicada(1:k));
            
            if (sum(porcentajeVarianzaExplicada(1:k)) >= umbralPorcentajeDeVarianza) && (numCompAdmitidos == 0)
                numCompAdmitidos = k;
            end
        end
        
        aux = Xtrain*coefCompPrincipales;
        Xtrain = aux(:,1:numCompAdmitidos);
        
        aux = Xtest*coefCompPrincipales;
        Xtest = aux(:,1:numCompAdmitidos);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        NumArboles=10;
        Modelo = TreeBagger(NumArboles,Xtrain,Ytrain);

        Yest = predict(Modelo,Xtest);
        Yest = str2double(Yest);
        
        EficienciaTest(fold) = sum(Ytest == Yest)/length(Ytest);
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    
elseif (punto == 5)
    rng('default');
    [matrizB,Info] = lassoglm(X,Y,'binomial','NumLambda',25,'CV',10);
    lassoPlot(matrizB,Info,'PlotType','CV');
    lassoPlot(matrizB,Info,'PlotType','Lambda','XScale','log');
end