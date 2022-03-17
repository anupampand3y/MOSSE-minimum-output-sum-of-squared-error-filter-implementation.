clc;
close all;
clear;
path='imgs3';
%% Video frames to pictures:(Only one time run this portion)
% 
% vid=VideoReader('ship.avi');
%   numFrames = vid.NumberOfFrames;
%   n=numFrames;
% for i = 1:1:n
%   frames = read(vid,i);
% 
%   imwrite(frames,['C:\Users\eranu\OneDrive - Indian Institute of Technology Bhubaneswar\Documents\MATLAB\ADSP LAB\Exp4\MOSSE Final\imgs3\imgs' num2str(i, '%04.f'), '.tif']);
% end

%% Reading Images
filenames=dir(fullfile(path,'*.tif'));
noi=numel(filenames);   %number of images
N=noi;     %No. of images
f=fullfile(path, filenames(1).name);
im = imread(f);
f = figure('Name', 'Select object to track'); imshow(im);

% for i=1:noi
%     f=fullfile(path, filenames(i).name);
%     im1 = imread(f);
%     imshow(im1);
% end

template = getrect;
close(f); clear f;
c1=template(2)+template(4)/2;
c2=template(1)+template(3)/2;


%% Preprocessing Step1: Convert template RGB to Gray
template_img = rgb2gray(im);
template_img = imcrop(template_img, template);
template_img=  mat2gray(template_img);
imshow(template_img);

%% Preprocessing Step2: Apply Log transformation
a = double(template_img)/255; %Normalized Image for log transformation
c = 300; % Constant
temp_log = c*log(1 + (a)); % Log Transform
imshow((temp_log)),title('Log Transformation Image');

%% Preprocessing Step3: Normalize to zero mean and Unit variance
temp_norm = (temp_log - mean(temp_log, 'all')) ./ std2(temp_log);
% mean(temp_norm,'all')
% std2(temp_norm)
% imshow((temp_norm))

%% Preprocessing Step4: Transform to Fourier Domain
Fi=fft2(temp_norm);
%imshow((abs(F)))

%% Synthetically target image(Ground truth)
sigma = 2;
g_size = size(im);      %Size of first frame
[xx,yy] = ndgrid(1:g_size(1), 1:g_size(2));
g=(exp(-((xx-c1).^2 + (yy-c2).^2)./(2*sigma)));
g = mat2gray(g);
g = imcrop(g, template); %
G = fft2(g);
height = size(g,1);
width = size(g,2);

%% MOSSE filter Initialization
epsilon=0.00001;   % To avoid zero by zero form
Ai = (G.*conj(Fi));
Bi = (Fi.*conj(Fi));
eta = 0.125;
%%
for i=1:N    
    f=fullfile(path, filenames(i).name);
    im = imread(f);
    if (size(im,3) == 3) %if RGB
        img = rgb2gray(im); %then convert to grayscale
    end
    if (i == 1) %For the first image
        Ai = eta.*Ai;
        Bi = eta.*Bi + epsilon;
    else
        Hi = Ai./Bi; % for second picture onwards, solve the filter H*
        %Preprocessing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fi = imcrop(img, template);
        fi=imresize(fi, [height width]);
        %Log transformation
        a = double(fi)/255; %Normalized Image for log transformation
        c = 300; % Constant
        temp_log = c*log(1 + (a)); % Log Transform
        %end of log transform
        %Normalize to zero mean and unit variance
        temp_norm = (temp_log - mean(temp_log,'all')) ./ std2(temp_log);
        Fi=fft2(temp_norm);
        %Updating template according to max peak location%%%%%%%%%%%%%%%%%%
        
        gi = uint8(255*mat2gray(ifft2(Hi.*Fi)));
        max_g = max(gi(:)); 
        [xxx, yyy] = find(gi == max_g); % Max g location
        dx = mean(xxx)-height/2;
        dy = mean(yyy)-width/2;
        template = [template(1)+dy template(2)+dx width height]; %update coordinates
        
        %Last updated Ai and Bi%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ai = eta.*(G.*conj(Fi)) + (1-eta).*Ai;
        Bi = eta.*(Fi.*conj(Fi)) + (1-eta).*Bi;
    end
    % Preview images after each iteration
    result = insertShape(im, 'Rectangle', template,'LineWidth',5);
    imshow(result);
end
close all;