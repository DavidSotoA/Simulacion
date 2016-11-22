clear all;
clc;

archivo=importdata('C:\Users\Usuario\Desktop\parkinsons_updrs.data');
datos=archivo.data;
Y1=datos(:,5);
Y2=datos(:,6);
X=datos(:,7:end);
%%%%%%%%%%%%%%%%%%%%%%%%pasar datos para clasificacion%%%%%%%%%%%%%%%%%%%%%
p2=prctile(Y1,33.33333);
p3=prctile(Y1,66.66666);

Y1(Y1<p2)=1;
Y1(Y1>=p2 & Y1<p3)=2;
Y1(Y1>=p3 & Y1<=(max(Y1)))=3;

p2=prctile(Y2,33.33);
p3=prctile(Y2,66.66);

Y2(Y2<p2)=1;
Y2(Y2>=p2 & Y2<p3)=2;
Y2(Y2>=p3 & Y2<=(max(Y2)))=3;

Y=[Y1 Y2];
selectFeaturesY1=[1 1 1 0 0 0 1 0 0 1 1 1 1 1 1 1];
%selectFeaturesY1=[0 1 0 0 1 0 1 0 0 0 1 0 1 0 1 1];
selectFeaturesY2=[1 1 1 0 0 1 0 1 0 0 0 1 1 1 1 1];
selectFeaturesY=[];
%% seleccion de caracteristiscas
 opciones = statset('display','iter');
    sentido = 'forward';
    
    [caracteristicasElegidas, proceso] = sequentialfs(@funcionForest,X,Y1,'direction',sentido,'options',opciones);
%% pearson y fisher
 alpha = 0.05;
 [correlacion,p]= corrcoef(X);
 
 indicesClase1 = find(Y1 == 1);
 indicesClase2 = find(Y1 == 2);
 indicesClase3 = find(Y1 == 3);
 
 mediaClase1 = mean(X(indicesClase1,:) ,1);
 mediaClase2 = mean(X(indicesClase2,:) ,1);
 mediaClase3 = mean(X(indicesClase3,:) ,1);
 
 media = [mediaClase1; mediaClase2;mediaClase3];
 
 varClase1 = var(X(indicesClase1,:) ,1);
 varClase2 = var(X(indicesClase2,:) ,1);
 varClase3 = var(X(indicesClase3,:) ,1);
    
 varianza = [varClase1;varClase2;varClase3];
 
 coef = zeros(1,16);
    for i=1:2
        for j=1:2
            if (j ~= i)
                numerador = (media(i,:) - media(j,:)).^2;
                denominador = varianza(i,:) + varianza(j,:);
                coef = coef + (numerador./denominador);
            end
        end
    end
%% extraccion de caracteristicas




%%
tic;
% close all
%addpath(genpath('netlab'));
%load('Data.mat');


%X=Data(:,1:6);
%Y=Data(:,end);
addpath(genpath('netlab'));
NumMuestras=size(X,1); 
Rept=10;

EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));

RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);

Mezclas=[1,2,3,4,5,6];
fid=fopen('datos.txt','w');
fprintf(fid, 'MEZCLAS GAUSSIANAS\n\n');
for m=1:2
    y=Y(:,m);
    fprintf(fid, '\nY%d\n',m);
    for k=1:3;
        if(k==1)
            fprintf(fid, 'matriz de covarianza esferica\n');
        elseif(k==2)
            fprintf(fid, 'matriz de covarianza diagonal\n');
        else
            fprintf(fid, 'matriz de covarianza completa\n');
        end
        for i=1:length(Mezclas)
            for fold=1:Rept

                %%% Se hace la partición de las muestras %%%
                %%%      de entrenamiento y prueba       %%%

                rng('default');
                particion=cvpartition(NumMuestras,'Kfold',Rept);
                indices=particion.training(fold);
                Xtrain=X(particion.training(fold),:);
                Xtest=X(particion.test(fold),:);
                Ytrain=y(particion.training(fold));
                Ytest=y(particion.test(fold));

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%% Se normalizan los datos %%%

                [XtrainNormal,mu,sigma] = zscore(Xtrain);
                XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%

                vInd=(Ytrain == 1);
                XtrainC1 = Xtrain(vInd,:);
                if ~isempty(XtrainC1)
                    Modelo1=entrenarGMM(XtrainC1,Mezclas(i),k);
                else
                    error('No hay muestras de todas las clases para el entrenamiento');
                end

                vInd=(Ytrain == 2);
                XtrainC2 = Xtrain(vInd,:);
                if ~isempty(XtrainC2)
                    Modelo2=entrenarGMM(XtrainC2,Mezclas(i),k);
                else
                    error('No hay muestras de todas las clases para el entrenamiento');
                end

                vInd=(Ytrain == 3);
                XtrainC3 = Xtrain(vInd,:);
                if ~isempty(XtrainC3)
                    Modelo3=entrenarGMM(XtrainC3,Mezclas(i),k);
                else
                    error('No hay muestras de todas las clases para el entrenamiento');
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%% Validación de los modelos. %%%

                probClase1=testGMM(Modelo1,Xtest);
                probClase2=testGMM(Modelo2,Xtest);
                probClase3=testGMM(Modelo3,Xtest);
                Matriz=[probClase1,probClase2,probClase3];

                [~,Yest] = max(Matriz,[],2);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                MatrizConfusion = zeros(NumClases,NumClases);
                for j=1:size(Xtest,1)
                    MatrizConfusion(Yest(j),Ytest(j)) = MatrizConfusion(Yest(j),Ytest(j)) + 1;
                end
                EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
                
                z1=sum(MatrizConfusion);
                z2=sum(MatrizConfusion,2);
                for j=1:NumClases
                    RecallTest(fold,j)= MatrizConfusion(j,j)/z1(j);
                    PrecisionTest(fold,j)=MatrizConfusion(j,j)/z2(j);
                end
                
            end
            Eficiencia = mean(EficienciaTest);
            
            Recall=zeros(1,NumClases);
            ICRecall=zeros(1,NumClases);
            Precision=zeros(1,NumClases);
            ICPrecision=zeros(1,NumClases);
            for j=1:NumClases
                Recall(j)=nanmean(RecallTest(:,j));
                ICRecall(j)=nanstd(RecallTest(:,j));
                Precision(j)=nanmean(PrecisionTest(:,j));
                ICPrecision(j)=nanstd(PrecisionTest(:,j));
            end
            
            ICEficiencia = std(EficienciaTest);
            Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
            disp(Texto);
            fprintf(fid, 'La eficiencia obtenida con %d mezclas fue de %f +- %f \n',Mezclas(i),Eficiencia,ICEficiencia);
           
            fprintf(fid, 'La presicion es:  \n');
            for j=1:NumClases
                fprintf(fid, 'clase %d: %f +- %f\n',j,Precision(j),ICPrecision(j));
            end
            fprintf(fid, 'el recall es:  \n');
            for j=1:NumClases
                fprintf(fid, 'clase %d: %f +-  %f\n',j,Recall(j),ICRecall(j));
            end
            
        end
        fprintf(fid, '\n');
    end
end
fclose(fid);
toc;
%%
%load('DatosClasificacion.mat');
%Xclas=Xclas(:,1:3);
%NumMuestras=size(Xclas,1); 

Rept=10;
EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
NumMuestras=size(X,1); 
vecinos=[1,2,3,4,5,6,7,8,9];
fid=fopen('datos.txt','a');
fprintf(fid, 'K VECINOS\n\n');
for m=1:2
    y=Y(:,m);
    fprintf(fid, '\nY%d\n',m);
    for i=1:length(vecinos)
        for fold=1:Rept
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=y(particion.training(fold));
            Ytest=y(particion.test(fold));

            [Xtrain,mu,sigma]=zscore(Xtrain);
            Xtest=normalizar(Xtest,mu,sigma);

            Yest=vecinosCercanos(Xtest,Xtrain,Ytrain,vecinos(i),'class'); 

            MatrizConfusion = zeros(NumClases,NumClases);
            for j=1:size(Xtest,1)
                MatrizConfusion(Yest(j),Ytest(j)) = MatrizConfusion(Yest(j),Ytest(j)) + 1;
            end
            EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
            z1=sum(MatrizConfusion);
            z2=sum(MatrizConfusion,2);
            for j=1:NumClases
               RecallTest(fold,j)= MatrizConfusion(j,j)/z1(j);
               PrecisionTest(fold,j)=MatrizConfusion(j,j)/z2(j);
            end

        end
        Eficiencia = mean(EficienciaTest);

        Recall=zeros(1,NumClases);
        ICRecall=zeros(1,NumClases);
        Precision=zeros(1,NumClases);
        ICPrecision=zeros(1,NumClases);
        for j=1:NumClases
            Recall(j)=nanmean(RecallTest(:,j));
            ICRecall(j)=nanstd(RecallTest(:,j));
            Precision(j)=nanmean(PrecisionTest(:,j));
            ICPrecision(j)=nanstd(PrecisionTest(:,j));
        end

        ICEficiencia = std(EficienciaTest);
        Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
        disp(Texto);
        fprintf(fid, 'La eficiencia obtenida con %d vecinos fue de %f +- %f \n',vecinos(i),Eficiencia,ICEficiencia);

        fprintf(fid, 'La presicion es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +- %f\n',j,Precision(j),ICPrecision(j));
        end
        fprintf(fid, 'el recall es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +-  %f\n',j,Recall(j),ICRecall(j));
        end

    end
end
fclose(fid);
%%
Rept=10;
EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y1));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
NumMuestras=size(X,1); 
neuronas=[10,20,30,40,50,60,70,80,100,150,200];
fid=fopen('datos.txt','a');
fprintf(fid, 'RNA\n\n');

for h=1:2
    a=Y(:,h);
    [m,s]=size(a);
    y1=zeros(m,3);

    for i=1:length(a)
        x=a(i);
        y1(i,x)=1;
    end
    y=y1;

    fprintf(fid, '\nY%d\n',h);
    for i=1:length(neuronas)
        for fold=1:Rept
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=y(particion.training(fold),:);
            [~,Ytest]=max(y(particion.test(fold),:),[],2);

            %%% Se normalizan los datos %%%

            [XtrainNormal,mu,sigma]=zscore(Xtrain);
            XtestNormal=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%
            Modelo=RNAClass(Xtrain,Ytrain,neuronas(i));

            Yest=testRNA(Modelo,Xtest);
            [~,Yest]=max(Yest,[],2);

            MatrizConfusion=zeros(NumClases,NumClases);
            for j=1:size(Xtest,1)
                MatrizConfusion(Yest(j),Ytest(j))=MatrizConfusion(Yest(j),Ytest(j)) + 1;
            end
            EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
            z1=sum(MatrizConfusion);
            z2=sum(MatrizConfusion,2);
            for j=1:NumClases
               RecallTest(fold,j)= MatrizConfusion(j,j)/z1(j);
               PrecisionTest(fold,j)=MatrizConfusion(j,j)/z2(j);
            end
        end
        Eficiencia = mean(EficienciaTest);

        Recall=zeros(1,NumClases);
        ICRecall=zeros(1,NumClases);
        Precision=zeros(1,NumClases);
        ICPrecision=zeros(1,NumClases);
        for j=1:NumClases
            Recall(j)=nanmean(RecallTest(:,j));
            ICRecall(j)=nanstd(RecallTest(:,j));
            Precision(j)=nanmean(PrecisionTest(:,j));
            ICPrecision(j)=nanstd(PrecisionTest(:,j));
        end

        ICEficiencia = std(EficienciaTest);
        Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
        disp(Texto);
        fprintf(fid, 'La eficiencia obtenida con %d neuronas fue de %f +- %f \n',neuronas(i),Eficiencia,ICEficiencia);

        fprintf(fid, 'La presicion es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +- %f\n',j,Precision(j),ICPrecision(j));
        end
        fprintf(fid, 'el recall es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +-  %f\n',j,Recall(j),ICRecall(j));
        end
         
    end
end
fclose(fid);
%%

%addpath(genpath('netlab'));
% load('Data.mat');
% 
% 
% X=Data(:,1:6);
% Y=Data(:,end);

NumMuestras=size(X,1); 
Rept=10;

EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
arboles=[10,20,30,40,50,60,70,80,100,300,400,500,700];
fid=fopen('datos.txt','w');
fprintf(fid, 'RANDOM FOREST\n\n');
for m=1:1
    y=Y(:,m);
    fprintf(fid, '\nY%d\n',m);
    for i=1:length(arboles)
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


            %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%
            Modelo=entrenarFOREST(arboles(i),Xtrain,Ytrain);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Validación de los modelos. %%%

            Yest=testFOREST(Modelo,Xtest);
            Yest = cell2mat(Yest);
            Yest = str2num(Yest);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            MatrizConfusion = zeros(NumClases,NumClases);
            for j=1:size(Xtest,1)
                MatrizConfusion(Yest(j),Ytest(j)) = MatrizConfusion(Yest(j),Ytest(j)) + 1;
            end
            EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
             z1=sum(MatrizConfusion);
            z2=sum(MatrizConfusion,2);
            for j=1:NumClases
               RecallTest(fold,j)= MatrizConfusion(j,j)/z1(j);
               PrecisionTest(fold,j)=MatrizConfusion(j,j)/z2(j);
            end

        end

        Eficiencia = mean(EficienciaTest);

        Recall=zeros(1,NumClases);
        ICRecall=zeros(1,NumClases);
        Precision=zeros(1,NumClases);
        ICPrecision=zeros(1,NumClases);
        for j=1:NumClases
            Recall(j)=nanmean(RecallTest(:,j));
            ICRecall(j)=nanstd(RecallTest(:,j));
            Precision(j)=nanmean(PrecisionTest(:,j));
            ICPrecision(j)=nanstd(PrecisionTest(:,j));
        end

        ICEficiencia = std(EficienciaTest);
        Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
        disp(Texto);
        fprintf(fid, 'La eficiencia obtenida con %d arboles fue de %f +- %f \n',arboles(i),Eficiencia,ICEficiencia);

        fprintf(fid, 'La presicion es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +- %f\n',j,Precision(j),ICPrecision(j));
        end
        fprintf(fid, 'el recall es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +-  %f\n',j,Recall(j),ICRecall(j));
        end
         
    end
end
fclose(fid);
%%

%load('DatosClasificacion.mat');
% 
 NumMuestras=size(X,1); 
 Rept=10;

EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
gamma=[1,1.5,2,2.5,3,4,5,6];
boxConstraint=[1,10,100,200,500,1000];
kernel=['linear','quadratic','polynomial'];
fid=fopen('datos.txt','a');
fprintf(fid, 'SMV\n\n');
for t=1:3
    fprintf(fid, kernel(t));
    for m=1:2
        y=Y(:,m);
        fprintf(fid, '\nY%d\n',m);
        for j=1:length(boxConstraint)
            for i=1:length(gamma)
                for fold=1:Rept

                    %%% Se hace la partición de las muestras %%%
                    %%%      de entrenamiento y prueba       %%%

                    rng('default');
                    particion=cvpartition(NumMuestras,'Kfold',Rept);
                    Xtrain=X(particion.training(fold),:);
                    Xtest=X(particion.test(fold),:);
                    Ytrain=y(particion.training(fold),:);
                    Ytest=y(particion.test(fold));

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %%% Se normalizan los datos %%%

                    [Xtrain,mu,sigma]=zscore(Xtrain);
                    Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %%% Entrenamiento de los modelos. Se usa la metodologia One vs All. %%%

                    %%% Complete el codigo implimentando la estrategia One vs All.
                    %%% Recuerde que debe de entrenar un modelo SVM para cada clase.
                    %%% Solo debe de evaluar las muestras con conflicto.

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    u=unique(Ytrain);
                    numClasses=length(u);
                    result = zeros(length(Xtest(:,1)),1);
                    resultados = zeros(size(Xtest,1),numClasses);
                    %build models
                    for k=1:numClasses
                        %Se entrena un modelo por clases para hacer one vs all
                        %G1vAll  tiene un 1 si la muestra es de la clase k, cero si
                        % es de otra clase
                        G1vAll=(Ytrain==u(k));
                        models(k) = svmtrain(Xtrain,G1vAll,'boxconstraint',boxConstraint(j),'kernel_function',kernel(t),'method','LS','rbf_sigma',gamma(i));
                    end

                    %Aqui realiza la prediccion por cada muestra de Xtest
                    % por cada modelo mira en cual la salida es 1
                    % No resuelve muestras con conflicto

                    for n=1:size(Xtest,1)

                        for k=1:numClasses

                            if(svmclassify(models(k),Xtest(n,:))==1) 
                                break;
                            end
                        end
                        result(n) = k;
                    end

                    % Aqui esta para que mire las muestras que tienen conflicto 
                    % Son las que tienen mas de un 1 en la misma fila

                    for k=1:numClasses
                       resultados(:,k) = svmclassify(models(k),Xtest);
                    end

                    Yest = result;

                    MatrizConfusion=zeros(NumClases,NumClases);
                    for k=1:size(Xtest,1)
                        MatrizConfusion(Yest(k),Ytest(k))=MatrizConfusion(Yest(k),Ytest(k)) + 1;
                    end
                    EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
                    z1=sum(MatrizConfusion);
                    z2=sum(MatrizConfusion,2);
                    for u=1:NumClases
                       RecallTest(fold,u)= MatrizConfusion(u,u)/z1(u);
                       PrecisionTest(fold,u)=MatrizConfusion(u,u)/z2(u);
                    end

                end
                Eficiencia = mean(EficienciaTest);

                Recall=zeros(1,NumClases);
                ICRecall=zeros(1,NumClases);
                Precision=zeros(1,NumClases);
                ICPrecision=zeros(1,NumClases);
                for u=1:NumClases
                    Recall(u)=nanmean(RecallTest(:,u));
                    ICRecall(u)=nanstd(RecallTest(:,u));
                    Precision(u)=nanmean(PrecisionTest(:,u));
                    ICPrecision(u)=nanstd(PrecisionTest(:,u));
                end

                ICEficiencia = std(EficienciaTest);
                Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
                disp(Texto);
                fprintf(fid, 'La eficiencia obtenida con %f gamma y boxcontraint de %f fue de %f +- %f \n',gamma(i),boxConstraint(j),Eficiencia,ICEficiencia);

                fprintf(fid, 'La presicion es:  \n');
                for u=1:NumClases
                    fprintf(fid, 'clase %d: %f +- %f\n',u,Precision(u),ICPrecision(u));
                end
                fprintf(fid, 'el recall es:  \n');
                for u=1:NumClases
                    fprintf(fid, 'clase %d: %f +-  %f\n',u,Recall(u),ICRecall(u));
                end
            end
        end
    end
end
fclose(fid);
%% mejores modelos

X = X(:, selectFeaturesY1==1);
%umbralPorcentajeDeVarianza = 85;
Rept=10;
EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
NumMuestras=size(X,1); 
vecinos=[9];
fid=fopen('datos.txt','w');
fprintf(fid, 'K VECINOS\n\n');
for m=1:1
    y=Y(:,m);
    fprintf(fid, '\nY%d\n',m);
    for i=1:length(vecinos)
        for fold=1:Rept
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=y(particion.training(fold));
            Ytest=y(particion.test(fold));

            Yest=vecinosCercanos(Xtest,Xtrain,Ytrain,vecinos(i),'class'); 

            MatrizConfusion = zeros(NumClases,NumClases);
            for j=1:size(Xtest,1)
                MatrizConfusion(Yest(j),Ytest(j)) = MatrizConfusion(Yest(j),Ytest(j)) + 1;
            end
            EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
            z1=sum(MatrizConfusion);
            z2=sum(MatrizConfusion,2);
            for j=1:NumClases
               RecallTest(fold,j)= MatrizConfusion(j,j)/z1(j);
               PrecisionTest(fold,j)=MatrizConfusion(j,j)/z2(j);
            end

        end
        Eficiencia = mean(EficienciaTest);

        Recall=zeros(1,NumClases);
        ICRecall=zeros(1,NumClases);
        Precision=zeros(1,NumClases);
        ICPrecision=zeros(1,NumClases);
        for j=1:NumClases
            Recall(j)=nanmean(RecallTest(:,j));
            ICRecall(j)=nanstd(RecallTest(:,j));
            Precision(j)=nanmean(PrecisionTest(:,j));
            ICPrecision(j)=nanstd(PrecisionTest(:,j));
        end

        ICEficiencia = std(EficienciaTest);
        Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
        disp(Texto);
        fprintf(fid, 'La eficiencia obtenida con %d vecinos fue de %f +- %f \n',vecinos(i),Eficiencia,ICEficiencia);

        fprintf(fid, 'La presicion es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +- %f\n',j,Precision(j),ICPrecision(j));
        end
        fprintf(fid, 'el recall es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +-  %f\n',j,Recall(j),ICRecall(j));
        end

    end
end
fclose(fid);

NumMuestras=size(X,1); 
Rept=10;
X = X(:, selectFeaturesY1==1);
EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
arboles=[500];
fid=fopen('datos.txt','a');
fprintf(fid, 'RANDOM FOREST\n\n');

for m=1:1
    y=Y(:,m);
    fprintf(fid, '\nY%d\n',m);
    for i=1:length(arboles)
        for fold=1:Rept

            %%% Se hace la partición de las muestras %%%
            %%%      de entrenamiento y prueba       %%%

            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=y(particion.training(fold));
            Ytest=y(particion.test(fold));
            
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


            %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%
            Modelo=entrenarFOREST(arboles(i),Xtrain,Ytrain);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Validación de los modelos. %%%

            Yest=testFOREST(Modelo,Xtest);
            Yest = cell2mat(Yest);
            Yest = str2num(Yest);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            MatrizConfusion = zeros(NumClases,NumClases);
            for j=1:size(Xtest,1)
                MatrizConfusion(Yest(j),Ytest(j)) = MatrizConfusion(Yest(j),Ytest(j)) + 1;
            end
            EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
             z1=sum(MatrizConfusion);
            z2=sum(MatrizConfusion,2);
            for j=1:NumClases
               RecallTest(fold,j)= MatrizConfusion(j,j)/z1(j);
               PrecisionTest(fold,j)=MatrizConfusion(j,j)/z2(j);
            end

        end

        Eficiencia = mean(EficienciaTest);

        Recall=zeros(1,NumClases);
        ICRecall=zeros(1,NumClases);
        Precision=zeros(1,NumClases);
        ICPrecision=zeros(1,NumClases);
        for j=1:NumClases
            Recall(j)=nanmean(RecallTest(:,j));
            ICRecall(j)=nanstd(RecallTest(:,j));
            Precision(j)=nanmean(PrecisionTest(:,j));
            ICPrecision(j)=nanstd(PrecisionTest(:,j));
        end

        ICEficiencia = std(EficienciaTest);
        Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
        disp(Texto);
        fprintf(fid, 'La eficiencia obtenida con %d arboles fue de %f +- %f \n',arboles(i),Eficiencia,ICEficiencia);

        fprintf(fid, 'La presicion es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +- %f\n',j,Precision(j),ICPrecision(j));
        end
        fprintf(fid, 'el recall es:  \n');
        for j=1:NumClases
            fprintf(fid, 'clase %d: %f +-  %f\n',j,Recall(j),ICRecall(j));
        end
         
    end
end
fclose(fid);
%%
NumMuestras=size(X,1); 
Rept=10;

X = X(:, selectFeaturesY2==1);
EficienciaTest=zeros(1,Rept);
NumClases=length(unique(Y));
RecallTest=zeros(Rept,NumClases);
PrecisionTest=zeros(Rept,NumClases);
gamma=[1];
boxConstraint=[10];

fid=fopen('datos.txt','a');
fprintf(fid, 'SMV\n\n');
for m=2:2
    y=Y(:,m);
    fprintf(fid, '\nY%d\n',m);
    for j=1:length(boxConstraint)
        for i=1:length(gamma)
            for fold=1:Rept

                %%% Se hace la partición de las muestras %%%
                %%%      de entrenamiento y prueba       %%%

                rng('default');
                particion=cvpartition(NumMuestras,'Kfold',Rept);
                Xtrain=X(particion.training(fold),:);
                Xtest=X(particion.test(fold),:);
                Ytrain=y(particion.training(fold),:);
                Ytest=y(particion.test(fold));

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%% Se normalizan los datos %%%

                [Xtrain,mu,sigma]=zscore(Xtrain);
                Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%% Entrenamiento de los modelos. Se usa la metodologia One vs All. %%%

                %%% Complete el codigo implimentando la estrategia One vs All.
                %%% Recuerde que debe de entrenar un modelo SVM para cada clase.
                %%% Solo debe de evaluar las muestras con conflicto.

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                u=unique(Ytrain);
                numClasses=length(u);
                result = zeros(length(Xtest(:,1)),1);
                resultados = zeros(size(Xtest,1),numClasses);
                %build models
                for k=1:numClasses
                    %Se entrena un modelo por clases para hacer one vs all
                    %G1vAll  tiene un 1 si la muestra es de la clase k, cero si
                    % es de otra clase
                    G1vAll=(Ytrain==u(k));
                    models(k) = svmtrain(Xtrain,G1vAll,'boxconstraint',boxConstraint(j),'kernel_function','rbf','method','LS','rbf_sigma',gamma(i));
                end

                %Aqui realiza la prediccion por cada muestra de Xtest
                % por cada modelo mira en cual la salida es 1
                % No resuelve muestras con conflicto

                for n=1:size(Xtest,1)

                    for k=1:numClasses

                        if(svmclassify(models(k),Xtest(n,:))==1) 
                            break;
                        end
                    end
                    result(n) = k;
                end

                % Aqui esta para que mire las muestras que tienen conflicto 
                % Son las que tienen mas de un 1 en la misma fila

                for k=1:numClasses
                   resultados(:,k) = svmclassify(models(k),Xtest);
                end

                Yest = result;

                MatrizConfusion=zeros(NumClases,NumClases);
                for k=1:size(Xtest,1)
                    MatrizConfusion(Yest(k),Ytest(k))=MatrizConfusion(Yest(k),Ytest(k)) + 1;
                end
                EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
                z1=sum(MatrizConfusion);
                z2=sum(MatrizConfusion,2);
                for u=1:NumClases
                   RecallTest(fold,u)= MatrizConfusion(u,u)/z1(u);
                   PrecisionTest(fold,u)=MatrizConfusion(u,u)/z2(u);
                end

            end
            Eficiencia = mean(EficienciaTest);

            Recall=zeros(1,NumClases);
            ICRecall=zeros(1,NumClases);
            Precision=zeros(1,NumClases);
            ICPrecision=zeros(1,NumClases);
            for u=1:NumClases
                Recall(u)=nanmean(RecallTest(:,u));
                ICRecall(u)=nanstd(RecallTest(:,u));
                Precision(u)=nanmean(PrecisionTest(:,u));
                ICPrecision(u)=nanstd(PrecisionTest(:,u));
            end

            ICEficiencia = std(EficienciaTest);
            Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(ICEficiencia)];
            disp(Texto);
            fprintf(fid, 'La eficiencia obtenida con %f gamma y boxcontraint de %f fue de %f +- %f \n',gamma(i),boxConstraint(j),Eficiencia,ICEficiencia);

            fprintf(fid, 'La presicion es:  \n');
            for u=1:NumClases
                fprintf(fid, 'clase %d: %f +- %f\n',u,Precision(u),ICPrecision(u));
            end
            fprintf(fid, 'el recall es:  \n');
            for u=1:NumClases
                fprintf(fid, 'clase %d: %f +-  %f\n',u,Recall(u),ICRecall(u));
            end
        end
    end
end
fclose(fid);
