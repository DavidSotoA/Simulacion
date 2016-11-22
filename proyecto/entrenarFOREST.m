function Modelo = entrenarFOREST(NumArboles,X,Y)
    rng(1); 
    Modelo = TreeBagger(NumArboles,X,Y,'Method','classification');
end