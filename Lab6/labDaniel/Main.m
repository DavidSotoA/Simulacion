clc
clear all
close all

punto=2;

load('DatosSeleccion.mat');

if (punto == 1)

    alpha = 0.05;
    [correlacion,p]= corrcoef([X,Y],'alpha',alpha);
    
	% Revisar en el workspace la matriz de correlacion
    
    %%%%%%%%%%%% Indice de Fisher
    
    indicesClase0 = find(Y == 0);
    indicesClase1 = find(Y == 1);
    
    mediaClase0 = mean(X(indicesClase0,:) ,1);
    mediaClase1 = mean(X(indicesClase1,:) ,1);
    
    media = [mediaClase0; mediaClase1];
    
    varClase0 = var(X(indicesClase0,:) ,1);
    varClase1 = var(X(indicesClase1,:) ,1);
    
    varianza = [varClase0;varClase1];
    
    %Se calcula el indice de Fisher
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
    
    coefN = coef./max(coef);
    
	% Revisar en el workspace el indice fisher original y el normalizado
	
    %%%%%%%%% Fin punto 1
    
elseif (punto == 2)
    
    Rept=10; % Se establece que el numero de pliegues (folds) para la validación cruzada
    NumMuestras=size(X,1); % Se determina cuantas son las muestras de entrenamiento
    EficienciaTest=zeros(1,Rept); % Se inicializa un vector columna en donde se guardara la eficiencia de los modelos en cada iteracion
    eleccion=[];
    % Se crean una variables de configuración para el proceso de seleccion de caracteriticas
    opciones = statset('display','final'); % Si desea ver los resultados en cada iteracion use 'iter', pero si solo desea ver el resultado final use 'final'
    sentido = 'forward'; % Use 'forward' para busqueda hacia adelante (opcion por defecto) o 'backward' para busqueda hacia atras
    
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
        
         [caracteristicasElegidas, ~] = sequentialfs(@funcionForest,Xtrain,Ytrain,'direction',sentido,'options',opciones);
         XReducidas = X(:, caracteristicasElegidas);
         eleccion=[eleccion,caracteristicasElegidas];
    end

end