%Analysing Map_Mapping Data in Matlab 

%Images are stored in the MapMapping, Post_Analysis Folder 
 %Then Stored by Experiment, version name, then by group 

%% Required Scripts 
  %ShadedErrorBars 
   %http://uk.mathworks.com/matlabcentral/fileexchange/26311-shadederrorbar
  
%% Reformatting the Images 

% For ICA Owen carries out the following filtering steps: 
 % Downsample size to 679*300*80  
 % Smooth with a Gaussian filter (alpha = 2) 
 % Sample the central brain into voxels 
 % Remove samples with many missing values and voxels with any missing
   %values 
 % Normalise each fish for brightness 
 
% The first two filtering steps can be achieved by loading
 % Map-Mapping Images from the Source folder (post-smoothing, down-sizing
  % etc)

%Load in Owen's Masks
 %This Contains: 
  %A logical stack mask of the whole brain 
  %A vectorised/voxel map of the whole brain 
tic
  load('Vector_Masks.mat')
 
%Open data of interest from the Post-analysis folder 
 %This should contain downsized and smoothed "Source" images

folder = dir2('C:\map_mapping\perk_exp\registration\MapMapping\Post_Analysis'); 

progressbar('Experiment','Version','Group','Sample') %Initialise progress bars
for e = 1:size(folder,1) %For each experiment
    
    version = dir2(strcat('C:\map_mapping\perk_exp\registration\MapMapping\Post_Analysis\'...
        ,folder(e).name)); %Open the experiment folder
    
    for v = 1:size(version,1) %For each version
        
        group = dir2(strcat('C:\map_mapping\perk_exp\registration\MapMapping\Post_Analysis\'...
            ,folder(e).name,'\',version(v).name)); %Open the version folder
        
        for g = 1:size(group,1) %For each group
            
            sample = dir2(strcat('C:\map_mapping\perk_exp\registration\MapMapping\Post_Analysis\'...
                ,folder(e).name,'\',version(v).name,'\',group(g).name)); %Open the group folder (Het, Hom, WT)
            
            %Read the Samples into doubles, cut the central brain out,
             %Convert to voxels and store linear pERK/tERK
             %vectors
            a = 1; %Start a counter
            for s = 1:2:size(sample,1) %For each sample 
                
                %Generate NaN Matracies in two channels 
                tERK(1:679,1:300,1:80) = NaN;  
                pERK(1:679,1:300,1:80) = NaN;  
                
                tERKVec(1:size(BrainPixelLabelsVec,1),1) = NaN; 
                pERKVec(1:size(BrainPixelLabelsVec,1),1) = NaN; 
                
                tERKInVoxels(1,1:max(BrainPixelLabelsVec)) = NaN; 
                pERKInVoxels(1,1:max(BrainPixelLabelsVec)) = NaN; 

                for slice = 1:80 %For each slice  
                tERK(:,:,slice) = double(imread(strcat('C:\map_mapping\perk_exp\registration\MapMapping\Post_Analysis\'...
                ,folder(e).name,'\',version(v).name,'\',group(g).name,'\',...
                sample(s).name),slice)); %Read the tERK sample
                
                pERK(:,:,slice) = double(imread(strcat('C:\map_mapping\perk_exp\registration\MapMapping\Post_Analysis\'...
                ,folder(e).name,'\',version(v).name,'\',group(g).name,'\',...
                sample(s+1).name),slice)); %Read the pERK sample
                end 
            
                tERK(tERK == 0) = NaN; %Make Zero Values = NaN (tERK)
                pERK(pERK == 0) = NaN; %Make Zero Values = NaN (pERK)
                
                tERKVec = tERK(BrainPixelLabelsKeep); %Mask/vectorise the signals (px)
                pERKVec = pERK(BrainPixelLabelsKeep); %Mask/vectorise the signals (px)
                
                %Convert to Voxels - this downsample reduces the data from 3 million
                 %pixels to 300,000 voxels
                 %Binning roughly 10px (mean) into every voxel
                tERKInVoxels(1,:) = accumarray(BrainPixelLabelsVec, tERKVec, [], @nanmean);
                pERKInVoxels(1,:) = accumarray(BrainPixelLabelsVec, pERKVec, [], @nanmean);
            
            %Store each channel seperately 
            tERK_data{e}{v}{g}(a,:) = tERKInVoxels;  
            pERK_data{e}{v}{g}(a,:) = pERKInVoxels; 
            
            %Store linear pERK/tERK vector 
            pERK_tERK_data{e}{v}{g}(a,:) = pERKInVoxels./tERKInVoxels;   
            a = a + 1; %Add to counter
            
            clear tERK pERK tERKVec pERKVec pERKInVoxels tERKInVoxels
             progressbar([],[],[],s/size(sample,1)); %Update sample progress bar 
            end
             
            clear sample 
         progressbar([],[],g/size(group,1)); %Update the group progress bar
        end
        
        clear group
     progressbar([],v/size(version,1)); %Update the version progress bar 
    end
    
  clear version
 progressbar(e/size(folder,1)); %Update the experiment progress bar   
end
  
clearvars -except tERK_data pERK_data  pERK_tERK_data BrainPixelLabelsVec BrainPixelLabelsKeep    
toc

%% Pre-analysis Filtering 

%Build a single matrix for the data 
all_data = []; 
for e = 1:size(pERK_tERK_data,2) % For each experiment 
    for v = 1:size(pERK_tERK_data{e},2) % For each version 
        for g = 1:size(pERK_tERK_data{e}{v},2) % For each group 
            all_data = [all_data ; pERK_tERK_data{e}{v}{g}];
        end 
    end 
end 

%Filter Data 

%Remove fish with many missing values 
%     FishWithNaN = logical(sum(isnan(scrap), 2)>3000); %If more than 8% of the voxels are missing
%     scrap(FishWithNaN, :) = []; %Remove these samples 

% Get rid of voxels that contain fish with NaNs
     %I.e. Remove columns in which any fish contains a NaN Value 
    VoxelsWithNaN = sum(isnan(all_data), 1)>0; %Creates a logical marking NaN Columns
     %Note this logical is particularly useful as you can use it later to
      %To crop other signals to match the dataset 
      
% Keep track of with Voxels Need to be filled back in
    VoxelsToReExpand = true(size(VoxelsWithNaN));
    VoxelsToReExpand(VoxelsWithNaN) = 0;
    
    %Remove Columns with NaN from the datasets   
    for e = 1:size(pERK_tERK_data,2) % For each experiment
        for v = 1:size(pERK_tERK_data{e},2) % For each version
            for g = 1:size(pERK_tERK_data{e}{v},2) % For each group
                pERK_tERK_data{e}{v}{g}(:,VoxelsWithNaN) = [];
                tERK_data{e}{v}{g}(:,VoxelsWithNaN) = [];
                pERK_data{e}{v}{g}(:,VoxelsWithNaN) = [];
            end
        end
    end
    
    all_data_filtered = all_data; % Remove these values from all_data  
    all_data_filtered(:,VoxelsWithNaN) = [];
    
%Normalise images by brightness 
% prcts = prctile(pERK_tERK_Matrix, 10, 2); % normalize each fish by dividing by 10th precentile of dataset
%     pERK_tERK_Matrix = pERK_tERK_Matrix./repmat(prcts, [1, size(pERK_tERK_Matrix,2)]); 
% 
%     meanVoxel = mean(pERK_tERK_Matrix, 1); % subtract the mean from each voxel
%     pERK_tERK_Matrix = pERK_tERK_Matrix - repmat(meanVoxel, [size(pERK_tERK_Matrix, 1), 1]);

%% Building in pERK/tERK Activity Patterns 

%Load in Vector masks 
load('C:\Users\Marcus\Documents\MATLAB\Z-Brain\Vector_Masks.mat')

%Load in the activity pattern 
 %Note - you need to split the channels first (1 = down, 2 = up)
 %Note - you need to convert the channel to an 8-bit image 
 %Note - you need to save the image as a tiff

[filename, pathname] = uigetfile('*.tif', 'Select a Significant Delta Medians file'); %Select a geno file
if isequal(filename,0) %If no file is selected
    error('No File Selected') %Show Error
else %If selected
    disp(['User selected ', fullfile(pathname, filename)]) %Show selected filename
end

%Pre-allocate 
activity_pattern(1:679,1:300,1:80) = NaN; 

%Read in activity pattern 
for slice = 1:80 %For each slice  
       activity_pattern(:,:,slice) = double(imread(strcat(pathname,... 
           filename),slice));          
end

clear filename pathname

%Keep unfiltered version 
activity_pattern_unfiltered = activity_pattern; 

%Filter and reformat
activity_pattern(activity_pattern == 0) = NaN; %Remove zero values 
activity_pattern_Vec = activity_pattern(BrainPixelLabelsKeep); %Mask/vectorise the signals (px)

%Now Convert the activity pattern from a vector to voxels 
activity_pattern_InVoxels(1,:) = accumarray(BrainPixelLabelsVec, activity_pattern_Vec, [], @nanmean);

activity_pattern_mask = activity_pattern_InVoxels > 0; %Generate Logical Mask from the activity pattern 

%Crop out the voxels with NaN that were removed from pERK_tERK_Matrix 
activity_pattern_mask(:,VoxelsWithNaN) = [];

%This mask can now be applied to pERK/tERK signals etc (see sections below) 

%% Voxel Analysis 

%Applying the mask 
for e = 1:size(pERK_tERK_data,2) % For each experiment
    for v = 1:size(pERK_tERK_data{e},2) % For each version
        for g = 1:size(pERK_tERK_data{e}{v},2) % For each group
            for s = 1:size(pERK_tERK_data{e}{v}{g},1) % For each sample 
                %pERK/tERK
                sample = pERK_tERK_data{e}{v}{g}(s,:);
                pERK_tERK_data_masked{e}{v}{g}(s,:) = sample(activity_pattern_mask);
                clear sample
                
                %tERK
                sample = tERK_data{e}{v}{g}(s,:);
                tERK_data_masked{e}{v}{g}(s,:) = sample(activity_pattern_mask);
                clear sample
                
                %pERK
                sample = pERK_data{e}{v}{g}(s,:);
                pERK_data_masked{e}{v}{g}(s,:) = sample(activity_pattern_mask);
                clear sample
            end
        end
    end
end

%% Figures 

colors(3,:) = [0.3922    0.5843    0.9294]; %WT - pale blue
colors(1,:) = [0.4392    0.5020    0.5647]; %Het - grey 
colors(2,:) = [1    0   0]; %Hom - red 

% Scattering 

% Build a single data structure + Anova Tags 
pERK_tERK_masked_str = []; 
experiment_tag = []; 
group_tag = []; 
a = 1; % Start a counter 
for e = 1:size(pERK_tERK_data_masked,2) % For each experiment
    for v = 1:size(pERK_tERK_data_masked{e},2) % For each version
        for g = 1:size(pERK_tERK_data_masked{e}{v},2) % For each group
            pERK_tERK_masked_str = [pERK_tERK_masked_str ; pERK_tERK_data_masked{e}{v}{g}];   
            experiment_tag = [experiment_tag ; e*ones(size(pERK_tERK_data_masked{e}{v}{g},1),1)]; 
            group_tag = [group_tag ; g*ones(size(pERK_tERK_data_masked{e}{v}{g},1),1)]; 
            group_sizes(a,1) = size(pERK_tERK_data_masked{e}{v}{g},1);  
            a = a + 1;
        end 
    end 
end 
pERK_tERK_masked_str_sum = nansum(pERK_tERK_masked_str,2); 

% Line Plot 
figure; hold on; 
subplot(2,1,2); hold on; 
for g = 1:max(group_tag) % For each group 
plot([1 2],...
    [nanmean(pERK_tERK_masked_str_sum(experiment_tag == 2 & group_tag == g)),...
    nanmean(pERK_tERK_masked_str_sum(experiment_tag == 1 & group_tag == g))],...
    'color',colors(g,:),'linewidth',3); 
errorbar(1,nanmean(pERK_tERK_masked_str_sum(experiment_tag == 2 & group_tag == g)),...
    nanstd(pERK_tERK_masked_str_sum(experiment_tag == 2 & group_tag == g))/...
   sqrt(size(pERK_tERK_masked_str_sum(experiment_tag == 2 & group_tag == g),1)),...
    'color',colors(g,:),'linewidth',3); 
errorbar(2,nanmean(pERK_tERK_masked_str_sum(experiment_tag == 1 & group_tag == g)),...
    nanstd(pERK_tERK_masked_str_sum(experiment_tag == 1 & group_tag == g))/...
   sqrt(size(pERK_tERK_masked_str_sum(experiment_tag == 1 & group_tag == g),1)),...
    'color',colors(g,:),'linewidth',3); 
end 

% Nice Figure 
set(gca,'Fontsize',32)
ylabel({'Total' ; 'pERK/tERK Signal'})
xlabel('Time (Hours)') 
set(gca,'XTick',[1:2])
set(gca,'XTickLabels',{'09:30' ; '22:30'})
clear scrap; scrap = ylim; 
axis([0.5 2.5 scrap(1) scrap(2)]); clear scrap;
%title({'Evening ^{-/-}' ; 'Activated Mask'})
set(gcf, 'InvertHardCopy', 'off'); % For Black Background

% Two Way Anova 
[p,h,stats] = anovan(pERK_tERK_masked_str_sum,...
    {experiment_tag, group_tag},'model','interaction',...
    'varnames',{'Time','Genotype'}); 

%% ICA 

% Pca 
X = pERK_tERK_data{1}{1}{3};  

% Rescale Data to 0 -> 1
% range = max(X) - min(X); % Calculate Range 
% for f = 1:size(X,2) % For each feature 
%     X(:,f) = X(:,f)  / range(f); % Rescale   
% end

% 
[coeff,score,latent,tsquared,explained,mu] = pca(X); 
[knee_dim] = knee_pt(explained) % Choose this many dimensions 

nComp = knee_dim; % Define the number of components 

T2 = pERK_tERK_data{1}{1}{1};

[icasig, A, W] = fastica(T2, 'approach', 'symm', 'numOfIC', nComp, 'firstEig', 1, 'lastEig', nComp, 'verbose', 'on');

%% Loading ICA Components

load('Vector_Masks.mat')

%Select a folder
folder_name = uigetdir;

%Open the folder
folder = dir2(folder_name);

progressbar('Sample') %Initialise progress bars
for s = 1:size(folder,1) %For each sample
    
    %Generate NaN Matracies in two channels
    channel_1(1:679,1:300,1:80) = NaN;
    
    %Read each tiff 
    for slice = 1:80 %For each slice
        channel_1(:,:,slice) = double(imread(strcat...
            (folder_name,'\',folder(s).name),slice)); %Read the tERK sample
    end;
    
    channel_1(channel_1 == 0) = NaN; %Make Zero Values = NaN (tERK)
    
    channel_1Vec = channel_1(BrainPixelLabelsKeep); %Mask/vectorise the signals (px)
    
    %Convert to Voxels - this downsample reduces the data from 3 million
    %pixels to 300,000 voxels
    %Binning roughly 10px (mean) into every voxel
    channel_1_InVoxels(1,:) = accumarray(BrainPixelLabelsVec, channel_1Vec, [], @nanmean);
    
    %Store each channel seperately
    channel_1_data(s,:) = channel_1_InVoxels;
        
    clear channel_1 channel_1Vec channel_1_InVoxels  
    progressbar(s/size(folder,1)) %Update sample progress bar
    
end

%% Activity Masking the ICA Components 
for s = 1:30
    s1 = ICA_channel_1_data(s,:);
    s2 = ICA_channel_2_data(s,:);
    
    ICA_channel_1_data_masked(s,:) = s1(activity_pattern_mask);
    ICA_channel_2_data_masked(s,:) = s2(activity_pattern_mask);
    
    clear s1 s2
end

%% Old Code 

n = [0 29 58 86]; %Hard Coded    

figure; hold on; 
for g = 1:size(pERK_tERK_data{1}{1},2); %For each group 
    line_colors = shadedErrorBar(1:size(pERK_tERK_activity_masked,2), ... 
        nanmean(pERK_tERK_activity_masked((n(g)+1):n(g+1),:)), ... 
         nanstd(pERK_tERK_activity_masked((n(g)+1):n(g+1),:)) ... 
          /sqrt((n(g+1)-n(g))), ...
           {'Color',[colors(g,:)]}); 
        
    legend_colors(g) = line_colors.mainLine; 
    
end; 

%Nice Figure; 
box off; 
set(gca,'Fontsize',44) 
xlabel('Voxel No.')
ylabel('pERK / tERK')
set(gca, 'Layer','top')
 %Normal
[h,icons,plots,str] = legend(legend_colors,'CHD8^{+/+} (n=29)','CHD8^{-/+} (n=29)','CHD8^{-/-} (n=28)','location','northwest')
 %Null Model 
%[h,icons,plots,str] = legend(legend_colors,...
%     'CHD8^{+/+} (n=19)','CHD8^{-/+} (n=18)','CHD8^{-/-} (n=18)',...
%     'CHD8^{+/+} Null Model','CHD8^{-/+} Null Model',...
%     'CHD8^{-/-} Null Model','location','northwest')

%Nice Legend
set(icons,'Linewidth',10)
set(icons,'FontSize',30)
legend('boxoff')

clear g line_colors legend_colors h icons plots str

% Testing a null model for the activity mask 

% Pre-allocation 
for g = 1:size(pERK_tERK_data{1}{1},2); %For each group
    null_model{g}(1:1000,1:sum(activity_pattern_mask)) = NaN; 
end; 

% Sampling
for g = 1:size(pERK_tERK_data{1}{1},2); %For each group 
    for t = 1:1000; %For 1000 times 
        %Reshape the pERK_tERK signal for the group into a vector
         %From this vector sample (without replacement) as many voxels 
          %As there are in the activity_pattern_mask 
        null_model{g}(t,:) = randsample(reshape...
            (pERK_tERK_Matrix((n(g)+1):n(g+1),:),[1,(n(g+1)-n(g))*size(pERK_tERK_Matrix,2)]) ...
            ,sum(activity_pattern_mask)); 
    end
end  

%Note it could be fairer to sample the same voxels from each data-set each
 %Repeat? 
%Not sure what the fairest model is? 
 
%Figure - null model as transparent dashed lines 
figure; hold on; 
for g = 1:size(pERK_tERK_data{1}{1},2); %For each group 
        
    %Null Models
    line_colors = shadedErrorBar(1:size(pERK_tERK_activity_masked,2), ... 
        nanmean(null_model{g}), ... 
         nanstd(null_model{g}) ... 
          /sqrt((n(g+1)-n(g))), ...
           {'r--','Color',[colors(g,:)]},1); 
     
        %legend_colors(g+size(pERK_tERK_data{1}{1},2)) = line_colors.mainLine; 

     %Activity Data
     line_colors = shadedErrorBar(1:size(pERK_tERK_activity_masked,2), ... 
        nanmean(pERK_tERK_activity_masked((n(g)+1):n(g+1),:)), ... 
         nanstd(pERK_tERK_activity_masked((n(g)+1):n(g+1),:)) ... 
          /sqrt((n(g+1)-n(g))), ...
           {'Color',[colors(g,:)]}); 
       
         legend_colors(g) = line_colors.mainLine; 

end; 


%% Generating a hollow activity mask 
% https://uk.mathworks.com/help/images/boundary-tracing-in-images.html

%Create a logical 3d mask from the activity pattern 
activity_pattern_3d_mask = logical(activity_pattern_unfiltered); 

%Fill in gaps and detect edges 
for s = 1:size(activity_pattern_3d_mask,3); %For each slice 
    activity_pattern_3d_mask_filled(:,:,s) = imfill(activity_pattern_3d_mask(:,:,s) ...
        , 'holes'); 
    
    boundaries{s,1} = bwboundaries(activity_pattern_3d_mask_filled(:,:,s)); 
    
end; 

% Create 3d Hollow Mask 

activity_pattern_hollow_mask(1:679,1:300,1:80) = 0; %Create a blank "stack"

for s = 1:size(activity_pattern_hollow_mask,3); %For each slice 
    if size(boundaries{s},1) > 0; %If there are boundaries  
        for r = 1:size(boundaries{s},1); %For each region
            for p = 1:size(boundaries{s}{r},1); %For each pixel in that region
                activity_pattern_hollow_mask(...
                    boundaries{s}{r}(p,1), boundaries{s}{r}(p,2),s) = 1;
                %Assign a value of 1 to a boundary pixel 
            end 
        end
    end 
end 

% Write Hollow Activity Mask to a Tiff Stack 
folder_name = uigetdir; 

for s = 1:size(activity_pattern_hollow_mask,3); %For each slice 
        
    imwrite(activity_pattern_hollow_mask(:,:,s),strcat(folder_name,'\','activity_pattern_hollow_mask.tif'),'WriteMode','append','Compression','none');

end;

%% Cut out an activity mask from a stack 

%Create a logical 3d mask from the activity pattern 
activity_pattern_3d_mask = logical(activity_pattern_unfiltered); 

%Load a stack (eg. an average stack) 
clear stack; stack(1:679,1:300,1:80) = NaN; 

[stack_name, stack_path] = uigetfile('*.tif','Select a Registerd Tiff Stack'); 

for slice = 1:80; %For each slice  
                stack(:,:,slice) = double(imread...
                    (strcat(stack_path,stack_name),slice)); %Read the sample            
end; 

stack(~activity_pattern_3d_mask) = 0; %Set values outside of the activity mask 
 %To Zero 
 
% Write cut out to tiff stack 
stack = uint16(stack); %Convert format

for s = 1:size(stack,3); %For each slice 
        
    imwrite(stack(:,:,s),strcat(stack_path,stack_name(1:end-4),'_','activity_pattern_cut.tif'),'WriteMode','append','Compression','none');

end;

%% Working 
% To overlay a hollow activity mask onto a stack (in this case scrap)
for s = 1:size(scrap,3); %For each slice
    subplot(8,10,s);
    hold on; 
    imshow(scrap(:,:,s)); 
    
    if size(boundaries{s},1) > 0; %If there are boundaries  
        for r = 1:size(boundaries{s},1); %For each region
            b = boundaries{s}{r};
            plot(b(:,2),b(:,1),'r','linewidth',3);
        end
    end 
    
end;

% Check to see that Intensity Projections from Cut out figures 
 % Show the data 
 
%Load in each cut out figure 
scrap = double(imread('C:\Users\Marcus\Documents\MBPhD\Presentations\Next Presentation\Hom_Channel_2_Cut_Activity_Pattern.tif'));
scrap(scrap == 0) = []; 

data(a,:) = scrap(:)'; 
a = a+1; 

colors(1,:) = [0.3922    0.5843    0.9294]; %WT - pale blue
colors(2,:) = [0.4392    0.5020    0.5647]; %Het - grey 
colors(3,:) = [1    0   0]; %Hom - red 

%tERK
figure; hold on; 
plot(data(3,:),'-','color',colors(2,:),'linewidth',3)
plot(data(5,:),'-','color',colors(3,:),'linewidth',3)
plot(data(1,:),'-','color',colors(1,:),'linewidth',3)

%pERK 
figure; hold on; 
plot(data(4,:),'-','color',colors(2,:),'linewidth',3)
plot(data(6,:),'-','color',colors(3,:),'linewidth',3)
plot(data(2,:),'-','color',colors(1,:),'linewidth',3)

%Nice Figure; 
box off; 
set(gca,'Fontsize',44) 
xlabel('Pixel No.')
ylabel('pERK')
set(gca, 'Layer','top')

%Line Plots 
figure; hold on; 
plot([1,2],[nanmean(data(4,:)),nanmean(data(3,:))],'color',colors(2,:),'linewidth',3)
plot([1,2],[nanmean(data(6,:)),nanmean(data(5,:))],'color',colors(3,:),'linewidth',3)
plot([1,2],[nanmean(data(2,:)),nanmean(data(1,:))],'color',colors(1,:),'linewidth',3)

%Nice Figure; 
box off; 
set(gca,'Fontsize',44) 
xlabel('Antibody')
ylabel('Mean Intensity')
set(gca, 'Layer','top')

