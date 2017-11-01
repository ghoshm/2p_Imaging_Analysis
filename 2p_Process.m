%% Processing 2photon Data for Map-Mapping 
 %This script batch proccess's data from our (Rihel) 2photon 
  %To make it compatible with the Z-brain map-mapping scripts 
   %Maps these stacks into the reference brain
    %Runs a modified version of Owen's MapMapping 
  
 %Required Software 
  %Matlab 
   %Note to write files to certain folders you'll need all permissions 
  %Fiji  
   %Set_MinMax.ijm 
    %This ImageJ Macro needs to be in the Fiji Pluggins folder 
  %Mij 
   %To set up (http://bigwww.epfl.ch/sage/soft/mij/) 
   %Add the Fiji App etc to your Matlab Path
    
   addpath(genpath('C:\Fiji.app'))
   Miji(false);
   %Note to process stacks in Miji you will need to increase your 
    %Java heap memory in Matlab 
     %http://uk.mathworks.com/matlabcentral/answers/92813-how-do-i-increase-the-heap-space-for-the-java-vm-in-matlab-6-0-r12-and-later-versions
  %Cygwin (If you are on a PC) 
  %CMTK 
  %Z-Brain Matlab Code 
  %ProgressBar 
   %http://uk.mathworks.com/matlabcentral/fileexchange/6922-progressbar
  %Nrrd Reader 
   %http://uk.mathworks.com/matlabcentral/fileexchange/34653-nrrd-format-file-reader/content/nrrdread.m
  %Dir2 
   %Marcus & Pedro 
   
 %Notes 

%--------------------------------------------------------------------------
 %-------------------------------------------------------------------------
 
 % 1. Pre-CMTK Processing - An alternative is to run TwoP_Process_Function 
    % This is more flexible as it allows for a choice of tERK channel and 
    % A choice of if the stacks should be flipped in Z or not 
  %Merge data from each channel into 2 stacks 
  %Data needs to be in the Two Photon (desktop), Pre-Process folder 
   %Inside this folder should be a folder for each experiment 
   %Inside each experiment folder should be a folder for each condition
    %Each should contain a folder for each fish - containing slices in seperate channels  
    %Note you'll need to remove single image folders 
    %Note that the script assumes two channels 
     %Later the CMTK registration (as coded here) specifies that channel 1
      %Is the tERK channel* 
    %Note that the script flips the stacks in Z (ie. a Dorso-ventral scran
     %Becomes a ventro-dorsal scan) 
      %Later the CMTK registration (to Z-Brain) specifies that the samples
       %Be ventro-dorsal*
   %As well as an Excel sheet of pixel width's for each stack
     %There needs to be a column for each group with a row for each fish
     %Indicating that fish's pixel width and height 
     %This must be named Widths.xlsx
   %* Note on the later points - these conditions are most easily fixed 
    %By running the script below and then applying corrections in ImageJ 
     %Afterwards 
    %Additionally stacks will have to be rotated to best match your
     %Reference stack pre-registration (make sure you enlarge the images to avoid cropping
      %Off portions)
     
     %This Goes Het, Hom, WT, 
tic
 folder = dir2('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process'); %Open the pre-process folder  

progressbar('Experiment','Group','Sample') %Initialise progress bars 
for f = 1:size(folder,1); %For each experiment  
    
    progressbar([],0) %Reset the Group Bar 
    
     files = dir2(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name)); %Open the experiment folder 
           pixel_widths = xlsread(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',folder(f).name,'/','Widths.xlsx')); %Read in the pixel widths for each stack

for g = 1:size(files,1)-1; %For each condition (the minus 1 removes the pixel width's file)    
    
    progressbar([],[],0) %Reset the Sample Bar 
        
     file_path = dir2(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name,'/',files(g).name)) %Open each group folder 

    for i = 1:size(file_path,1); %For every sample in each group  
         
        %Matlab
        stack_path = dir2(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name,'/',files(g).name,'/',file_path(i).name)); %Set the path for each sample 
         
        for c = 1:2; %For each channel (note this assumes a two channel image 
            
        for s = 5:((size(stack_path,1)-4)/2)+4; %Ignore the random files at the start and build a stack from the first channel 
            if c == 1;
         ZSeries{c,1}(:,:,s-4) = imread(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name,'/',files(g).name,'/',file_path(i).name,'/',stack_path(s).name));
            else 
         ZSeries{c,1}(:,:,s-4) = imread(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name,'/',files(g).name,'/',file_path(i).name,'/',stack_path(s+((size(stack_path,1)-4)/2)).name));
            end; 
        end; 
        
        %Flip this channel along z (assumes input stack is from dorsal ->
         %ventral)
        ZSeries{c,1} = flip(ZSeries{c},3); 
        
         %Noise Filtering by Offset Determination 
          ZSeries_Os{c} = double(ZSeries{c}(:)); %Linearise the image (variable as a double)
          Cmode{c} = mode(ZSeries_Os{c}); %Find the mode of the lineraised image 
          ZSeries_Os{c}(ZSeries_Os{c} > Cmode{c}) = []; %Remove values greater than the mode 
          ZSeries_Os{c} = ZSeries_Os{c} - Cmode{c}; %Subtract the mode 
          ZSeries_Os{c} = [ZSeries_Os{c}; -ZSeries_Os{c}]; %Reflect around the y axis 
          Cnoise{c} = floor(Cmode{c} + 2*(nanstd(ZSeries_Os{c}))); %Determine offset  
          ZSeries{c} = ZSeries{c} - Cnoise{c}; %Subtract the noise from the image
         
        %Write channels to tiff stacks 
        for k = 1:size(ZSeries{c},3); %Write channels to tiff stacks
            if c == 1;
          imwrite(ZSeries{c}(:,:,k),strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name,'/',files(g).name,'/',file_path(i).name,'/','ZSeries_Ch1.tif'),'WriteMode','append','Compression','none');
            else; 
          imwrite(ZSeries{c}(:,:,k),strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/', folder(f).name,'/',files(g).name,'/',file_path(i).name,'/','ZSeries_Ch2.tif'),'WriteMode','append','Compression','none');
            end; 
        end; 
        
        %Miji - open each stack, fill in properties and save as Nrrd files
         %Flexible Line
         if c ==1;             
            MIJ.run('Open...', strcat('path=[C:\\Users\\Marcus\\Desktop\\Two_Photon\\Pre-Process\\',folder(f).name,'\\',files(g).name,'\\',file_path(i).name,'\\','ZSeries_Ch1.tif]')); %Open the Ch1_stack 
            MIJ.run('Properties...',horzcat('channels=1 ','slices=',num2str(size(ZSeries{c,1},3)), ' frames=1 unit=um ','pixel_width=',num2str(pixel_widths(i,g)),' pixel_height=',num2str(pixel_widths(i,g)),' voxel_depth=2')); %Assign appropriate properties to each stack
            MIJ.run('Nrrd ... ',strcat('nrrd=C:/map_mapping/perk_exp/registration/images/',folder(f).name,'_',files(g).name,'_',file_path(i).name,'_','01.nrrd'));
         else
            MIJ.run('Open...', strcat('path=[C:\\Users\\Marcus\\Desktop\\Two_Photon\\Pre-Process\\',folder(f).name,'\\',files(g).name,'\\',file_path(i).name,'\\','ZSeries_Ch2.tif]')); %Open the Ch1_stack 
            MIJ.run('Properties...',horzcat('channels=1 ','slices=',num2str(size(ZSeries{c,1},3)), ' frames=1 unit=um ','pixel_width=',num2str(pixel_widths(i,g)),' pixel_height=',num2str(pixel_widths(i,g)),' voxel_depth=2')); %Assign appropriate properties to each stack
            MIJ.run('Nrrd ... ',strcat('nrrd=C:/map_mapping/perk_exp/registration/images/',folder(f).name,'_',files(g).name,'_',file_path(i).name,'_','02.nrrd'));
         end;
         MIJ.run('Close');
        end; 
        
         clear stack_path ZSeries ZSeries_Os Cmode Cnoise  
         progressbar([],[],i/size(file_path,1)); %Update Sample progress bar 
    end;   
         clear file_path 
         progressbar([],g/(size(files,1)-1)); %Update the Group progress bar 
end;
         clear files pixel_widths
         progressbar(f/size(folder,1)); %Update the Experiment progressbar
end;

%Move the Experiment Folders from the Pre-process to the Post Process
 %Folder 
for f = 1:size(folder,1); %For each experiment
    movefile(strcat('C:\Users\Marcus\Desktop\Two_Photon\Pre-Process\',... 
        folder(f).name),'C:\Users\Marcus\Desktop\Two_Photon\Processed'); 
end; 

toc

%Roughly a minute per brain (assuming ~150 slices per channel)  

%--------------------------------------------------------------------------
 %-------------------------------------------------------------------------

%2. Cygwin -> CMTK Registration 
%3 AML
% tic
% cygwindir = 'C:/cygwin64/bin/bash'; %Define the Cygwin Directory 
% commandline = ['"cd ' 'C:/map_mapping/perk_exp/registration; ' '"' 'C:/Fiji.app/bin/cmtk/munger" -b "C:/Fiji.app/bin/cmtk" -a -w -r 0102 -l af -X 52 -C 8 -G 80 -R 3 -A ''--accuracy 0.4 --auto-multi-levels 3'' -W ''--accuracy 1.6''  -T 4 -s "C:/Users/Marcus/Documents/MATLAB/Z-Brain/Ref20131120pt14pl2.nrrd" images"']; %Input the Registration command and parameters 
% system([cygwindir ' --login -c ' commandline]) %Pass these commands to Cygwin 
% toc

%4 AML 
tic
cygwindir = 'C:/cygwin64/bin/bash'; %Define the Cygwin Directory 
commandline = ['"cd ' 'C:/map_mapping/perk_exp/registration; ' '"' 'C:/Fiji.app/bin/cmtk/munger" -b "C:/Fiji.app/bin/cmtk" -a -w -r 0102 -l af -X 52 -C 8 -G 80 -R 3 -A ''--accuracy 0.4 --auto-multi-levels 4'' -W ''--accuracy 1.6''  -T 4 -s "C:/Users/Marcus/Documents/MATLAB/Z-Brain/Ref20131120pt14pl2.nrrd" images"']; %Input the Registration command and parameters 
system([cygwindir ' --login -c ' commandline]) %Pass these commands to Cygwin 
toc

%Timing
 %All Samples (38brains) from 160509 & 160516 took 31980s (8hrs, 53mins) to run
  %This approxiamtes to 14minutes per brain (16GB RAM)
  
%Accuracy
 
 %"Common" Registration problems + Potential Filters 
  %Mapping to the spine - read in a middle slice (74)
   %and ask if there are more colored pixels in the top 3rd of the image? 
%--------------------------------------------------------------------------
 %-------------------------------------------------------------------------

 %3. Preparing Stacks for Map-Mapping 
  
  source_files = dir2('C:\map_mapping\perk_exp\registration\reformatted'); %Open the reformatted folder 

  progressbar('Sample') %Initialise progress bars 
  s = 1; %Sample = 1
  for i = 2:4:size(source_files,1); %For each warped sample
      
     %Implement "PrepareStacksForMAPMapping" 
      MIJ.run('Open...', strcat('path=[C:\\map_mapping\\perk_exp\\registration\\reformatted\\',source_files(i).name,']')); %Open warped sample i 
       MIJ.run('Size...', 'width=300 depth=80 constrain average interpolation=Bilinear'); %Re-size the image
        MIJ.run('Gaussian Blur...', 'sigma=2 stack'); %Apply a blur  
      
     %Calculate the optimum max for the t-ERK channel 
      stack = MIJ.getCurrentImage; %Read the pixel values into a matrix 
       vector = double(stack(:)'); %Linearise this matrix 
        maximum(s,1) = prctile(vector,99.9); %Calculate the 0.1th percentile of the pixel intensities
       
        clear stack vector %Clean up the variables  
     
     %Calculate the optimum max from the p-ERK channel 
      MIJ.run('Open...', strcat('path=[C:\\map_mapping\\perk_exp\\registration\\reformatted\\',source_files(i+2).name,']')); %Open warped sample i+2 (the p-ERK channel for i) 
       MIJ.run('Size...', 'width=300 depth=80 constrain average interpolation=Bilinear'); %Re-size the image
        MIJ.run('Gaussian Blur...', 'sigma=2 stack'); %Apply a blur  
      
     %Calculate the optimum max for the p-ERK channel 
      stack = MIJ.getCurrentImage; %Read the pixel values into a matrix 
       vector = double(stack(:)'); %Linearise this matrix 
        maximum(s,2) = prctile(vector,99.9); %Calculate the 0.1th percentile of the pixel intensities
 
        if max(maximum(s,:)) < 255 %If the max is less than 255 - avoid "stretching the pixel intensities"
            max_cutoff = 255; %Set the max to 255
        else max_cutoff = max(maximum(s,:));
        end;  
       
        %Process the p-ERK Channel
        MIJ.run('Set MinMax', strcat('max=',num2str(max_cutoff))); %Adjust the max of the brightness to this value 
          MIJ.run('8-bit'); %Convert to 8-bit
           MIJ.run('Save',horzcat('Tiff..., path=','C:/map_mapping/perk_exp/registration/MapMapping/Source/',source_files(i+2).name(1:end-5),'_GauSmooth.tiff')); %Save as a Tiff 
              MIJ.run('Close'); %Close the Image
        
        %Process the t-ERK Channel 
        MIJ.run('Set MinMax', strcat('max=',num2str(max_cutoff))); %Adjust the max of the brightness to this value 
         MIJ.run('8-bit'); %Convert to 8-bit
          MIJ.run('Save',horzcat('Tiff..., path=','C:/map_mapping/perk_exp/registration/MapMapping/Source/',source_files(i).name(1:end-5),'_GauSmooth.tiff')); %Save as a Tiff 
              MIJ.run('Close'); %Close the Image
        
        clear stack vector max_cutoff %Clean up the variables  
                 
   progressbar(s/size(2:4:size(source_files,1),2)); %Update the Sample progressbar
   s = s+1; %Update the sample number for the progress bar 
   
  end; 
 
%Check Maximum Figure 
figure;
plot(maximum(:,1)','linewidth',3)
hold on; 
plot(maximum(:,2)','linewidth',3,'color','r')
plot((maximum(:,1)')-(maximum(:,2)'),'linewidth',3,'color','g')

%--------------------------------------------------------------------------
 %-------------------------------------------------------------------------

 %Building Nice Figures From MapMapping/VBrainViewer Data 
 
 %Load in the Z-Brain Variables 
 load('C:\Users\Marcus\Documents\MATLAB\Z-Brain\MaskDatabase.mat')
 
 %Choose the Variable that you are intrested in and find it's number in
  %MaskDatabaseNames (for example {'Diencephalon - Hypothalamus Hcrt
   %Neurons'} = Column number 23 
    %Note Changing your reference style in Excel to numerical column
     %Headings is helpful
  %Now Reshape the column you are interested in to match the Z-brain's
   %dimensions
    
 data = double(reshape(full(MaskDatabaseOutlines(:,61)), [height, width, Zs]));
 
 %Write this to a tif for use in Fiji
for i = 1:138; 
   imwrite(data(:,:,i),'C:\Users\Marcus\Desktop\preopticotb.tif','WriteMode','append','Compression','none');
end 

%--------------------------------------------------------------------------
 %-------------------------------------------------------------------------
 
 
 %Old Working 
 
%Reading the Tif images into a matrix
matrix(1:512,1:512,1:80) = NaN;

for i = 1:length(files);
    filename = files(i).name; 
    matrix(1:512,1:512,i) = imread(filename);
end; 


  % empirical determination of offset
            ts=double(im(:));
            gm.noise(p).mode=mode(ts); % find mode: should be center of noise distribution
            ts(ts>gm.noise(p).mode)=[]; % remove all values greater than mode
            ts=ts-gm.noise(p).mode; % subtract the mode, to give portion of noise distribution less then zero
            ts=[ts;-ts]; % reflect that distribution about y axis
            gm.noise(p).std=nanstd(ts); % determine its standard deviation
            gm.noise(p).empoffset=floor(gm.noise(p).mode+0*gm.noise(p).std); % offset is mode plus 2 standard deviations, rounded to lower integer
            [p gm.noise(p).empoffset]
            
        % subtract offset from im
        im=im-gm.noise(p).empoffset;
      
 %Old Mij/Miji Code
  %This is to run Imagej through Matlab (not required) 
    %javaaddpath 'C:\Program Files\MATLAB\R2014a\java\mij.jar'; 
    %javaaddpath 'C:\Program Files\MATLAB\R2014a\java\ij.jar';
    %MIJ.start; 
   %This is to run Fiji through Matlab (http://imagej.net/Miji):
    %javaaddpath 'C:\Fiji.app\jars\mij-1.3.6-fiji2.jar';
    %javaaddpath 'C:\Fiji.app\jars\ij-1.50g.jar';
    
    %javaaddpath 'C:\Program Files\MATLAB\R2014a\java\ij-1.50g.jar' ;
    %javaaddpath 'C:\Program Files\MATLAB\R2014a\java\mij-1.3.6-fiji2.jar'
    
    %Pedro's Cygwin Code 
cygwindir = 'C:/cygwin64/bin/bash';
regdir = 'C:/map_mapping/perk_exp/registration';
mungerdir = 'C:/Fiji.app/bin/cmtk/munger'; %Munger Directory 
cmtkdir = 'C:/Fiji.app/bin/cmtk'; %CMTK Directory 
refdir = 'C:/Users/Marcus/Documents/MATLAB/Z-Brain/Ref20131120pt14pl2.nrrd';
fltdir = 'images';
transfs = '-a -w -r'; %Kinds of Transformations (affine, warp and reformat) 
channels = '0102'; %Two channels 
outputt = 'af';
cores = 8;
metric = 'nmi';
coarsest = -1;
accuracy = 0.1;
exploration = 40;

fcmd = ['"cd ' regdir '; ' '"' mungerdir '" ' '-b "' cmtkdir '" ' transfs ' ' channels ' -l ' outputt ...
    ' -A ''' '--' metric ' --coarsest ' num2str(coarsest) ' -a ' num2str(accuracy) ' -e ' ...
    num2str(exploration) ''' -T ' num2str(cores) ' -s ' '"' refdir '"' ' ' fltdir '"'];

system([cygwindir ' --login -c ' commandline])

%Smoothing Working 
imfinfo('C:\Users\Marcus\Desktop\Test_Image.tif');
for s = 1:size(imfinfo('C:\Users\Marcus\Desktop\Test_Image.tif'),1); 
    scrap(:,:,s) = uint16(imread('C:\Users\Marcus\Desktop\Test_Image.tif',s));
end; 

array = double(scrap(:)');
hist(double(scrap(:)'));

tic
for i = min(array):max(array); 
    found(i+1) = size(find(array==i),2);
end;

plot(sums_pile)
hold on 
plot([find(sums_pile < 0.1,1),find(sums_pile < 0.1,1)],[0,100],'k')

toc

found_percentage = (found./size(array,2))*100;

for i = 1:size(found_percentage,2)
    sums_pile(i) = sum(found_percentage(1,i:end));  
end;  

load('C:\Users\Marcus\Desktop\Smoothing_Working.mat')

%Old Smoothing Working 
 %Old 
       %Find the maximum cut-off value 
     [stack] = nrrdread(strcat('C:\map_mapping\perk_exp\registration\reformatted\',source_files(i).name)); %Open the nrrd stack
      vector = double(stack(:)'); %Linearise the nrrd stack 
       maximum = prctile(vector,99.9); %Find the 0.1th percentile of the pixel intensities
       
  %Smoothing Working (very slow compared to the code above)
for s = 1:size(imfinfo('C:\Users\Marcus\Desktop\Test_Image-1.tif'),1); 
    scrap(:,:,s) = uint16(imread('C:\Users\Marcus\Desktop\Test_Image-1.tif',s));
end; 

array = double(scrap(:)');
hist(double(scrap(:)'));

tic
for i = min(array):maximum(array); 
    found(i+1) = size(find(array==i),2);
end;

found_percentage = (found./size(array,2))*100;

for i = 1:size(found_percentage,2)
    sums_pile(i) = sum(found_percentage(1,i:end));  
end;  

plot(sums_pile,'b','linewidth',3)
hold on 
plot([find(sums_pile < 0.1,1),find(sums_pile < 0.1,1)],[0,100],'--k','linewidth',3)
plot([4096,4096],[0, 100],'--k','linewidth',3)

%Nice Figure 
box off; 
axis([0,4500,0,100])
set(gca,'fontsize',44)
xlabel('Pixel Intensity')
ylabel('Frequency')
toc

load('C:\Users\Marcus\Desktop\Smoothing_Working.mat')