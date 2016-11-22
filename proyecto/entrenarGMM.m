function modelo = entrenarGMM(X,NumeroMezclas,MC)

    inputDim=size(X,2); %%%%% Numero de caracteristicas de las muestras
    if(MC==1)
      modelo = gmm(inputDim, NumeroMezclas,'spherical'); %% generar la estructura del modelo con matriz de covarianza esferica
    elseif(MC==2)
      modelo = gmm(inputDim, NumeroMezclas, 'diag'); %% generar la estructura del modelo con matriz de covarianza diagonal
    elseif(MC==3)
        modelo = gmm(inputDim, NumeroMezclas, 'full'); %% generar la estructura del modelo con matriz de covarianza completa
    end
    
    options = foptions;	
    modelo = gmminit(modelo, X, options); %% inicializar medias y matrices de covarianza con k-means
    
    options = zeros(1, 18);
    options(14) = 50;		% Max. Number of iterations.
    modelo = gmmem(modelo, X, options); %%entrenar el modelo
end
