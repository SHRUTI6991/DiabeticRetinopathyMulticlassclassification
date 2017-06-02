clear;

% read in data
load features;
load features1;



% init training data
classified = zeros(32,size(features,1));
group = zeros(size(features,1),1);

training_data = zeros(size(features,1),25);
experiment_data = zeros(size(features1,1),25);
for i=1:size(features,1)
   training_data(i,1) = features(i,1); 
   training_data(i,2) = features(i,2); 
   training_data(i,3) = features(i,3); 
   training_data(i,4) = features(i,4); 
   training_data(i,5) = features(i,5); 
   training_data(i,6) = features(i,6); 
   training_data(i,7) = features(i,7); 
   training_data(i,8) = features(i,8);
   training_data(i,9) = features(i,9); 
   training_data(i,10) = features(i,10); 
   training_data(i,11) = features(i,11); 
   training_data(i,12) = features(i,12);
   training_data(i,13) = features(i,13); 
   training_data(i,14) = features(i,14); 
   training_data(i,15) = features(i,15); 
   training_data(i,16) = features(i,16);
   training_data(i,17) = features(i,17); 
   training_data(i,18) = features(i,18); 
   training_data(i,19) = features(i,19); 
   training_data(i,20) = features(i,20);
   training_data(i,21) = features(i,21); 
   training_data(i,22) = features(i,22); 
   training_data(i,23) = features(i,23); 
   training_data(i,24) = features(i,24);
   training_data(i,25) = features(i,25); 
  
   
   experiment_data(i,1) =  features1(i,1); 
   experiment_data(i,2) =  features1(i,2); 
   experiment_data(i,3) =  features1(i,3); 
   experiment_data(i,4) =  features1(i,4); 
   experiment_data(i,5) =  features1(i,5); 
   experiment_data(i,6) =  features1(i,6); 
   experiment_data(i,7) =  features1(i,7); 
   experiment_data(i,8) =  features1(i,8); 
   experiment_data(i,9) =  features1(i,9); 
   experiment_data(i,10) =  features1(i,10); 
   experiment_data(i,11) =  features1(i,11); 
   experiment_data(i,12) =  features1(i,12); 
   experiment_data(i,13) =  features1(i,13); 
   experiment_data(i,14) =  features1(i,14); 
   experiment_data(i,15) =  features1(i,15); 
   experiment_data(i,16) =  features1(i,16); 
   experiment_data(i,17) =  features1(i,17); 
   experiment_data(i,18) =  features1(i,18); 
   experiment_data(i,19) =  features1(i,19); 
   experiment_data(i,20) =  features1(i,20); 
   experiment_data(i,21) =  features1(i,21); 
   experiment_data(i,22) =  features1(i,22); 
   experiment_data(i,23) =  features1(i,23); 
   experiment_data(i,24) =  features1(i,24); 
   experiment_data(i,25) =  features1(i,25); 
  
  
   group(i) = features(i,21);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(1,:) = svmclassify(SVMModel, experiment_data).';



% save classification results
dlmwrite('SVM Classification Result.txt',classified(1,:),'newline','pc','delimiter','')



