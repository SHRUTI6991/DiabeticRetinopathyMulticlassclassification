%% Extract optic disc and artifacts from one image
tic
% Read image
retinaRGB = imread('C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/train1/10_left.jpeg');
% Resize image
retinaRGB = resizeretina(retinaRGB, 752, 500);
% Get optic disc mask
closingThresholdValue = 0.64;
opticDiscDilationSize = 4;
artifactMinSize = 1100;
[opticDiscMask, artifactsMask] = getopticdiscartifacts(retinaRGB, ...
            closingThresholdValue, opticDiscDilationSize, artifactMinSize);
toc

%% Extract exudates from one image
tic
fileName = 'C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/train1/10_left.jpeg';    
% Read image
retinaRGB = imread(fileName);
% Resize image
retinaRGB = resizeretina(retinaRGB, 752, 500);
% Read optic disc mask
opticDiscMask = imread(fileName);
artifactsMask = imread(fileName);
% Get optic disc mask
opticDiscDilation = 10;
I = double(retinaRGB) / 255;
    I = sum(I, 3) ./ 3;
    % subplot(1, 2, 2), imshow(I); title('Intensity');

    %% Median filter on intensity channel
    % subplot(1, 2, 1), imshow(I); title('Before median filter');
    I = medfilt2(I);
    % subplot(1, 2, 2), imshow(I); title('Median filter on intensity channel');

    %% Histogram equalization
    % subplot(1, 2, 1), imshow(I); title('Before histogram equalization');
    I = adapthisteq(I);
    % subplot(1, 2, 2), imshow(I); title('Histogram equalization');    

    %% Remove vessels by grayscale closing
    % subplot(1, 2, 1), imshow(I); title('Before grayscale closing');
    se = strel('disk', 8);
    closeI = imclose(I, se);
    % subplot(1, 2, 2), imshow(closeI); title('Grayscale closing');

    %% Local standard deviation of an image
    % subplot(1, 2, 1), imshow(closeI); title('Before standard deviation');
    deviation = stdfilt(closeI, ones(7)); 
    % subplot(1, 2, 2), imshow(deviation, []); title('Standard deviation');

    %% Threshold and dilation
    % subplot(1, 2, 1), imshow(deviation, []); title('Before threshold and dilation');
    level = graythresh(deviation);
    mask = im2bw(deviation, level);
    se = strel('disk', 6);
    mask = imdilate(mask, se);
    % subplot(1, 2, 2), imshow(mask); title('Thresholded and dilated');

    %% Create region of interest
    retinaMask = im2bw(I, 0.2);
    retinaMask = imfill(retinaMask, 'holes');
    se = strel('disk', 16);
    retinaMask = imerode(retinaMask, se);
    % subplot(1, 2, 1), imshow(I), title('Original image');
    % subplot(1, 2, 2), imshow(I, 'InitialMag', 'fit')
    % Make a truecolor all-green image.
    % green = cat(3, zeros(size(I)), ones(size(I)), zeros(size(I)));
    % hold on
    % h = imshow(green);
    % hold off
    % Use our influence map as the AlphaData for the solid green image.
    % set(h, 'AlphaData', retinaMask)

    %% Remove circular shape around retina
    % subplot(1, 2, 1), imshow(mask); title('Before removing circular shape around');
    maskOfCenter = mask .* retinaMask;
    % subplot(1, 2, 2), imshow(maskOfCenter); title('Only region of interest');
    
    %% Flood fill
    % subplot(1, 2, 1), imshow(maskOfCenter); title('Before filling');
    maskFilled = imfill(maskOfCenter, 'holes');
   
    % subplot(1, 2, 2), imshow(maskFilled); title('Flood filled');

    %% Remove optic disc
    % subplot(1, 2, 1), imshow(maskFilled); title('Before optic disc elimination');
    se = strel('disk', opticDiscDilation);
    opticDiscMask = imdilate(opticDiscMask, se);
    opticDiscMask = rgb2gray(opticDiscMask);
    opticDiscMask = resizeretina(opticDiscMask, 752, 500);
    maskOfInterest = uint8(maskFilled) .* imcomplement(opticDiscMask);
    % subplot(1, 2, 2), imshow(maskOfInterest); title('Without optic disc');

    %% Remove artifacts
    % subplot(1, 2, 1), imshow(maskOfInterest); title('Before artifacts elimination');
    se = strel('disk', opticDiscDilation);
    artifactsMask = imdilate(artifactsMask, se);
    artifactsMask = rgb2gray(artifactsMask);
    artifactsMask  = resizeretina(artifactsMask, 752, 500);
    maskOfInterest = uint8(maskOfInterest) .* imcomplement(artifactsMask);
    % subplot(1, 2, 2), imshow(maskOfInterest); title('Without artifacts');
    
    %% Overlay mask on the original image
    % subplot(1, 2, 1), imshow(I); title('Before overlay');
    marker = I .* imcomplement(double(maskOfInterest));
    % subplot(1, 2, 2), imshow(marker); title('Overlay');

    %% Reconstruction
    % subplot(1, 2, 1), imshow(marker); title('Before reconstruction');
    reconstructed = imreconstruct(marker, I);
    % subplot(1, 2, 2), imshow(reconstructed); title('Reconstruction');

    %% Threshold on image differences
    diff = I - reconstructed;
    % subplot(1, 2, 1), imshow(diff, []), title('Difference before threshold');
    % level = graythresh(diff)
    level = 0.01;
    exudatesMask = im2bw(diff, level);
    % subplot(1, 2, 2), imshow(exudatesMask), title('Mask');

    %% Overlay exudates mask on the original image
    subplot(1, 2, 1), imshow(retinaRGB), title('Original image');
    subplot(1, 2, 2), imshow(I, 'InitialMag', 'fit')
    % Make a truecolor all-green image.
    green = cat(3, zeros(size(I)), ones(size(I)), zeros(size(I)));
    hold on
    h = imshow(green);
    hold off
    % Use our influence map as the AlphaData for the solid green image.
    set(h, 'AlphaData', exudatesMask)

%% Postprocess one image
tic
fileName = 'C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/train1/10_left.jpeg';  % 225_left
% Read images
retina = imread(fileName);
exudates = imread(fileName);
% Postprocessing
exudatesMaxSize = 160;
retina = resizeretina(retina, 752, 500);
retina = double(retina) / 255;
retina = sum(retina, 3) ./ 3;

 measurements = regionprops(exudates, 'Area');
    allAreas = [measurements.Area];
    exudatesValues = find(allAreas < exudatesMaxSize);
    %labeledExudates = bwlabel(exudates);
    %exudatesPostprocessed = ismember(labeledExudates, exudatesValues);

    opticDisc = imread(fileName);
redLesions = imread(fileName);

opticDisc = im2bw(opticDisc, 0.1);
redLesions = im2bw(redLesions, 0.1);

featuresExudates = getlesionsfeatures(exudates, opticDisc);

featuresRedLesions = getlesionsfeatures(redLesions, opticDisc);
% 12. Optic disc distance from center
opticDistance = getopticdistance(opticDisc);

% List all images
directory = 'C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/test1';
filesLeft = dir(strcat(directory, '*_left.jpeg'));
filesRight = dir(strcat(directory, '*_right.jpeg'));
files = [filesLeft; filesRight];
% For each image
nFiles = length(files);
for i = 25000 : nFiles
    fileName = strcat(directory, files(i).name);
    fprintf('Exudates, processing image %i / %i, %s.\n', i, nFiles, fileName);
    
    % Read image
    retinaRGB = imread(fileName);
    % Resize image
    retinaRGB = resizeretina(retinaRGB, 752, 500);
    % Read optic disc mask
    opticDiscMask = imread(fileName);
    artifactsMask = imread(fileName);
    % Get optic disc mask
    opticDiscDilation = 10;
    exudatesMask = getexudates(retinaRGB, opticDiscMask, artifactsMask, opticDiscDilation);
    
   
    
end
names1 = importdata('C:\Users\SHRUTI\Desktop\Project\MAJOR PROJECT\trainLabels\trainLabels2.xlsx');
% Initialize features matrix
nImages1 = 51;
nFeatures1 = 25;
features1 = zeros(nImages, nFeatures);
 
% For each image
for i = 2 : nImages1
   
    
    if (mod(i, 100) == 0)
        fprintf('Features test, processing image %i / %i, %s.\n', i, nImages1, fileName);
    end
    
    % Read images
    exudates = imread(strcat('C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/test1/', names1.textdata{i}, '.jpeg'));
    opticDisc = imread(strcat('C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/test1/', names1.textdata{i}, '.jpeg'));
    redLesions = imread(strcat('C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/test1/', names1.textdata{i}, '.jpeg'));
    
    % Make the images logicalp
    opticDisc = im2bw(opticDisc, 0.1);
    redLesions = im2bw(redLesions, 0.1);
    
    % Features 1 - 11 for exudates
    featuresExudates = getlesionsfeatures(exudates, opticDisc);
    % Features 1 - 11 (12 - 22) for red lesions
    featuresRedLesions = getlesionsfeatures(redLesions, opticDisc);
    % 12. Optic disc distance from center
    opticDistance = getopticdistance(opticDisc);
    
    % Save features into matrix and names intro array
    features1(i,:) = [featuresRedLesions, featuresExudates, opticDistance];
  
    
end
 
save('features1.mat','features1');
% Write csv file with features
csvwrite('C:/Users/SHRUTI/Desktop/Project/MAJOR PROJECT/features_test.csv', features1);
 

 