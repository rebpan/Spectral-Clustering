clear;
close all;

addpath(genpath(pwd));

S = load('.\Datasets\jain.txt');
true_labels = load('.\Datasets\jain_label.txt');
k = length(unique(true_labels));

scale_sig = 1;
tmp = squareform(pdist(S,'squaredeuclidean'))/(2*scale_sig^2);
A = exp(-tmp);

% figure,imshow(affinity,[]),title('affinity matrix');

D = diag(sum(A,2));

L = (D^-0.5)*A*(D^-0.5);

% select the largest eigenvectors
[eig_vecs,eig_vals] = eig(L);
[~,sort_index] = sort(diag(eig_vals),'descend');
X = eig_vecs(:,sort_index(1:k));

Y = X./repmat(sqrt(sum(X.^2,2)),1,k);
% figure,plot(Y(:,1),Y(:,2),'ro'),title('Data points after map'),grid on;

result_labels = kmeans(Y,k);
draw(S,result_labels);
