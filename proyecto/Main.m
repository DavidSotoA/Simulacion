%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
archivo=importdata('C:\Users\Usuario\Desktop\parkinsons_updrs.data');
datos=archivo.data;
Y1=datos(:,5);
Y2=datos(:,6);
Y=datos(:,5:6);
X=datos(:,7:end);
%%
    %%% Se crean los datos de forma aleatoria %%%
    rng('default');
    
    X1=linspace(-20,20,500); %% genera 500 datos espaciados en (x2-x1)/(n-1) entre -20 y 20
    X2=linspace(-50,50,500); %% genera 500 datos espaciados en (x2-x1)/(n-1) entre -50 y 50
    X=[X1',X2']; %%X contiene la transpuesta de X1 y X2
    
    %%%%%%%%%%%%%% generar dato de salida%%%%%%%%%%%%%%%%%%%%%%%%%%
    X=zscore(X); 
    Y=2*X.^3 + 4*X.^2 - 8*X + 5;
    Y=min(abs(Y),[],2) + max(abs(Y),[],2).*0.2.*randn(500,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    X=[X1',X2'];
   
%%% Se crean los datos de forma aleatoria %%%
    
    X1=linspace(-20,20,500); %% genera 500 datos espaciados en (x2-x1)/(n-1) entre -20 y 20
    X2=linspace(-50,50,500); %% genera 500 datos espaciados en (x2-x1)/(n-1) entre -50 y 50
    X=[X1',X2']; %%X contiene la transpuesta de X1 y X2
    
    %%%%%%%%%%%%%% generar dato de salida%%%%%%%%%%%%%%%%%%%%%%%%%%
    X=zscore(X); 
    Y=2*X.^3 + 4*X.^2 - 8*X + 5;
    Y=min(abs(Y),[],2) + max(abs(Y),[],2).*0.2.*randn(500,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    X=[X1',X2'];

    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%regresion multiple%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
Rept=10;
ECMTest=zeros(1,Rept);
eta=0.1;
grado=[1,2,3,4,5];
NumMuestras=size(X,1);
for i=1:length(grado)
    for fold=1:Rept
        %%% Se cambia el grado del polinomio %%%
        Xi=potenciaPolinomio(X,grado(i));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

        %%% Se hace la particion entre los conjuntos de entrenamiento y prueba.
        %%% Esta particion se hace forma aletoria %%%

        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        Xtrain=Xi(particion.training(fold),:);
        Xtest=Xi(particion.test(fold),:);
        Ytrain=Y1(particion.training(fold));
        Ytest=Y1(particion.test(fold));

        [r_train,~]=size(Xtrain);
        [r_test,~]=size(Xtest);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Normalizacion %%%
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);

        %%%%%%%%%%%%%%%%%%%%%

        %%% Se extienden las matrices %%%
        Xtrain=[Xtrain,ones(r_train,1)];
        Xtest=[Xtest,ones(r_test,1)];

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se aplica la regresion multiple %%%
        W=regresionMultiple(Xtrain,Ytrain,eta); %%% Se obtienen los W coeficientes del polinomio

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se encuentra el error cuadratico medio %%%
        Yesti=(W'*Xtest')';
        ECMTest(fold)=(sum((Yesti-Ytest).^2))/length(Ytest);
    end

    ECM = mean(ECMTest);
    IC = std(ECMTest);
    Texto=['El error cuadratico medio obtenido con grado ',num2str(i),' fue = ', num2str(ECM),' +- ',num2str(IC)];
    disp(Texto);
end
toc


%%
load('DatosRegresion.mat');
X=Xreg(:,1:6);
Y=Yreg;

%%
tic
Rept=10;
ECMTest=zeros(1,Rept);
NumMuestras=size(X,1);
h=[10,1,0.1,0.5,0.05];
%%%%%%%%%%%%%%%%%%%%%%%%%ventana de parzen%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(h)
    for fold=1:Rept
        rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold));
            Ytest=Y(particion.test(fold));

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);

        Yesti=ventanaParzen(Xtest,Xtrain,Ytrain,h(i),'regress');
        
        ECMTest(fold)=(sum((Yesti-Ytest).^2))/length(Ytest);
    end
    
    ECM = mean(ECMTest);
    IC = std(ECMTest);
    Texto=['El error cuadratico medio obtenido con una ventana de suavizado de ',num2str(i),' fue = ', num2str(ECM),' +- ',num2str(IC)];
    disp(Texto);
end
toc


%%
tic
Rept=2;
ECMTest=zeros(1,Rept);
NumMuestras=size(X,1);
neuronas=[10,20,30,40,50,100];
for i=1:length(neuronas)
    for fold=1:Rept
        rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold));
            Ytest=Y(particion.test(fold));

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        
        Modelo=RNA(Xtrain,Ytrain,neuronas(i));
        
        Yest=testRNA(Modelo,Xtest);
        
        ECMTest(fold)=(sum((Yest-Ytest).^2))/length(Ytest);
    end
    
    ECM = mean(ECMTest);
    IC = std(ECMTest);
    Texto=['El error cuadratico medio obtenido con  ',num2str(i),' neuronas fue = ', num2str(ECM),' +- ',num2str(IC)];
    disp(Texto);
end
