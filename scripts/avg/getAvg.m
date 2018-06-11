function Q = getAvg(data_file,model, output_file)

tic;
P=load(model);
P=P';
[dim, n_words] = size(P);

X = load(data_file);
X = spconvert(X);
%X(X>0)=1;
%X(X>1.38)=1; %idf threshold - testing TODO
%X(X>0.27 & X<=1.38)=0.8;
%X(X>0.13 & X<=1.27)=0.6;
%X(X>0.02 & X<=0.13)=0.4;
%X(X>0.01 & X<=0.02)=0.2;
%X(X>0.001 & X<=0.01)=0.01;


if size(X, 1) < n_words
 X(n_words,1) = 0;
end

[~, n_docs] = size(X);

Q=P*X;
L=sum(X);
L(find(L==0))=1;
for k = 1:n_docs %normalize by length
    Q(:,k) = Q(:,k)/L(k);
end

fprintf('done in %f seconds\n', toc);

Q=Q';
dlmwrite(output_file,Q, 'delimiter',' ');

end
