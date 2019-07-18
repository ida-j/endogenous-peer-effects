function f = normr(g)

n=size(g,1);

for i=1:n
    if sum(g(i,:))==0
        g(i,:)=0;
    else
    g(i,:)=g(i,:)/sum(g(i,:));
    end
end

f=g;
end