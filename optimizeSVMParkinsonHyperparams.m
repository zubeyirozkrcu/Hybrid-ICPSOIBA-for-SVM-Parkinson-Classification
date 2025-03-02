%In the Parkinson's disease dataset, it trains the SVM model with the RBF kernel, 
%takes the C (BoxConstraint) and gamma (KernelScale) hyperparameters from x(1,1) and x(1,2) and 
%calculates the average accuracy, sensitivity, precision, F1 score and specificity with 5-fold cross validation.
%Whichever metric you want to use should be written in the return value of the function.
%accuracyRate for Accuracy, sensitivityRate for Sensitivity,
%precisionRate for Precision, f1Score for f1 Score and specificityRate for Specificity.
function accuracyRate = optimizeSVMParkinsonHyperparams(x)
X = importdata('parkinsondisease.csv');
a = X.data;
c = a(:,1:16); %Eðitim kýsmýnda kullanýlacak feature'lar alýnýyor.
d = a(:, 18:23);
xdata = [c d];
group = a(:,17); %Hasta mý deðil mi.
k=5;
 
cvFolds = crossvalind('Kfold', group, k);  %# get indices of 5-fold CV

clear accRate;  
clear sensitivity;
clear precision; 
clear f1;
clear specificity;    
clear accuracyRate;
clear sensitivityRate;
clear precisionRate;
clear f1Score;
clear specificityRate;


for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

 
svmModel = fitcsvm(xdata(trainIdx,:), group(trainIdx), ...
                'KernelFunction', 'rbf', ...
                'KernelScale', x(1,2), ...
                'BoxConstraint', x(1,1), ...
                'Standardize', true);
				
	
    [pred, decValues] = predict(svmModel, xdata(testIdx,:));
    cm = confusionmat(group(testIdx),pred);
    accRate(i) = (cm(1,1) + cm(2,2)) / (cm(1,1) + cm(1,2) + cm(2,1) + cm(2,2));
    sensitivity(i) = (cm(1,1)) / (cm(1,1) + cm(2,1));  %recall, duyarlýlýk, hassasiyet
    precision(i) = (cm(1,1)) / (cm(1,1) + cm(1,2));     %kesinlik
    f1(i) = 2 * ((precision(i)*sensitivity(i)) / (precision(i) + sensitivity(i)));
    secicilik(i) = cm(2,2) / (cm(1,2) + cm(2,2));
end

accuracyRate = mean(accRate);
sensitivityRate = mean(sensitivity);
precisionRate = mean(precision);
f1Score = mean(f1);
specificityRate = mean(specificity);
end

