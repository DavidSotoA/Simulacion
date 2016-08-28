function dist = distancia(X1, X2)
    [~,N]=size(X1);
    
    sum=0;
    for i=1:N
       sum=sum+(X2(i)-X1(i)).^2; 
    end
    
    dist=sqrt(sum);

end

