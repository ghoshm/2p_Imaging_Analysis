%% Processing 2photon Data for Map-Mapping 
 %This script batch proccess's data from our (Rihel) 2photon 
  %To make it compatible with the Z-brain map-mapping scripts 
  
 %Required Software 
  %Matlab 
   %Note to write files to certain folders you'll need all permissions 
  %Fiji  
   %Set_MinMax.ijm 
    %This ImageJ Macro needs to be in the Fiji Pluggins folder 
  %Mij 
   %To set up (http://bigwww.epfl.ch/sage/soft/mij/) 
   %Add the Fiji App etc to your Matlab Path
   %Note to process stacks in Miji you will need to increase your 
    %Java heap memory in Matlab 
     %http://uk.mathworks.com/matlabcentral/answers/92813-how-do-i-increase-the-heap-space-for-the-java-vm-in-matlab-6-0-r12-and-later-versions
  %ProgressBar 
   %http://uk.mathworks.com/matlabcentral/fileexchange/6922-progressbar
  %Dir2 
   %Marcus & Pedro 

 
%% Pre-CMTK Processing 
  %Merge data from each channel into 2 stacks 
  %Data needs to be in the Two Photon (desktop), Pre-Process folder 
   %Inside this folder should be a folder for each experiment 
   %Inside each experiment folder should be a folder for each condition
    %Each should contain a folder for each fish - containing slices in seperate channels  
    %Note you'll need to remove single image folders 
    %Note that the script assumes two channels 
     %Later the CMTK registration specifies that channel 1
      %Is the tERK channel - thus you need to specify which channel is the tERK channel 
      %In the first function and this will be saved as channel 1
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
          
function [] = TwoP_Process_Function(tERK_Channel,Flip) 
        % Input arguments 
         % tERK_Channel - specify the samples tERK Channel as either 1 or 2
         % Flip - specify is channels should be flipped in Z (1) or not (0)
     
tic 
   addpath(genpath('C:\Fiji.app'))
   Miji(false);
   
% Check that the tERK Channel is specified as either channel 
if tERK_Channel ~= 1 && tERK_Channel ~= 2 
   msg = 'Specify tERK As Channel 1 or 2';
   error(msg)
end 

% Check that the flip is specified  
if tERK_Channel ~= 1 && tERK_Channel ~= 2 
   msg = 'Specify if samples should be flipped in Z (0 = no, 1 = yes)';
   error(msg)
end 

 folder = dir2('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process'); %Open the pre-process folder  

progressbar('Experiment','Group','Sample') %Initialise progress bars 

for f = 1:size(folder,1) %For each experiment  
    
    progressbar([],0) %Reset the Group Bar 
    
     files = dir2(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
         folder(f).name)); %Open the experiment folder 
     pixel_widths = xlsread(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
         folder(f).name,'/','Widths.xlsx')); %Read in the pixel widths for each stack
           
for g = 1:size(files,1)-1 %For each condition (the minus 1 removes the pixel width's file)    
    
    progressbar([],[],0) %Reset the Sample Bar 
        
     file_path = dir2(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
         folder(f).name,'/',files(g).name)); %Open each group folder 

    for i = 1:size(file_path,1) %For every sample in each group  
         
        %Matlab
        stack_path = dir2(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
            folder(f).name,'/',files(g).name,'/',file_path(i).name)); %Set the path for each sample 
         
        for c = 1:2 % For each Channel (note this assumes two channels) 
            
        for s = 5:((size(stack_path,1)-4)/2)+4 %Ignore the random files at the start and build a stack from channel c
            if c == 1
         ZSeries{c,1}(:,:,s-4) = imread(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/'...
             , folder(f).name,'/',files(g).name,'/',file_path(i).name,'/',stack_path(s).name));
            else 
         ZSeries{c,1}(:,:,s-4) = imread(strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
             folder(f).name,'/',files(g).name,'/',file_path(i).name,'/',stack_path(s+((size(stack_path,1)-4)/2)).name));
            end; 
        end; 
        
        % Flip the samples if requested 
        if Flip == 1
            ZSeries{c,1} = flip(ZSeries{c,1},3); 
        end 
        
         %Noise Filtering by Offset Determination 
          ZSeries_Os{c} = double(ZSeries{c,1}(:)); %Linearise the image (variable as a double)
          Cmode{c} = mode(ZSeries_Os{c}); %Find the mode of the lineraised image 
          ZSeries_Os{c}(ZSeries_Os{c} > Cmode{c}) = []; %Remove values greater than the mode (signal) 
          ZSeries_Os{c} = ZSeries_Os{c} - Cmode{c}; %Subtract the mode 
          ZSeries_Os{c} = [ZSeries_Os{c}; -ZSeries_Os{c}]; %Reflect around the y axis 
          Cnoise{c} = floor(Cmode{c} + 2*(nanstd(ZSeries_Os{c}))); %Determine offset  
          ZSeries{c,1} = ZSeries{c,1} - Cnoise{c}; %Subtract the noise from the image
         
        % Write channels to tiff stacks 
        for k = 1:size(ZSeries{c,1},3) % Write channels to tiff stacks
            if c == tERK_Channel % If the Current Channel is the tERK Channel
          imwrite(ZSeries{c,1}(:,:,k),strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
              folder(f).name,'/',files(g).name,'/',file_path(i).name,'/','ZSeries_Ch1.tif'),'WriteMode','append','Compression','none');
            else 
          imwrite(ZSeries{c,1}(:,:,k),strcat('C:/Users/Marcus/Desktop/Two_Photon/Pre-Process/',...
              folder(f).name,'/',files(g).name,'/',file_path(i).name,'/','ZSeries_Ch2.tif'),'WriteMode','append','Compression','none');
            end; 
        end; 
        
        % Miji - open each stack, fill in properties and save as Nrrd files

        if c == tERK_Channel % If the Current Channel is the tERK Channel             
            MIJ.run('Open...', strcat('path=[C:\\Users\\Marcus\\Desktop\\Two_Photon\\Pre-Process\\',...
                folder(f).name,'\\',files(g).name,'\\',file_path(i).name,'\\','ZSeries_Ch1.tif]')); % Open the tERK Stack 
            MIJ.run('Properties...',horzcat('channels=1 ','slices=',num2str(size(ZSeries{c,1},3)),...
                ' frames=1 unit=um ','pixel_width=',num2str(pixel_widths(i,g)),' pixel_height=',num2str(pixel_widths(i,g)),' voxel_depth=2')); % Assign appropriate properties to each stack
            MIJ.run('Nrrd ... ',strcat('nrrd=C:/map_mapping/perk_exp/registration/images/',...
                folder(f).name,'_',files(g).name,'_',file_path(i).name,'_','01.nrrd'));
            
        else
             
            MIJ.run('Open...', strcat('path=[C:\\Users\\Marcus\\Desktop\\Two_Photon\\Pre-Process\\',...
                folder(f).name,'\\',files(g).name,'\\',file_path(i).name,'\\','ZSeries_Ch2.tif]')); % Open the Ch1_stack 
            MIJ.run('Properties...',horzcat('channels=1 ','slices=',num2str(size(ZSeries{c,1},3)),...
                ' frames=1 unit=um ','pixel_width=',num2str(pixel_widths(i,g)),' pixel_height=',num2str(pixel_widths(i,g)),' voxel_depth=2')); % Assign appropriate properties to each stack
            MIJ.run('Nrrd ... ',strcat('nrrd=C:/map_mapping/perk_exp/registration/images/',...
                folder(f).name,'_',files(g).name,'_',file_path(i).name,'_','02.nrrd'));
         
         end;
         MIJ.run('Close');
        end; 
        
         clear stack_path ZSeries ZSeries_Os Cmode Cnoise  
         progressbar([],[],i/size(file_path,1)); % Update Sample progress bar 
    end;   
         clear file_path 
         progressbar([],g/(size(files,1)-1)); % Update the Group progress bar 
end;
         clear files pixel_widths
         progressbar(f/size(folder,1)); % Update the Experiment progressbar
end;

% Move the Experiment Folders from the Pre-process to the Post Process
 % Folder 
for f = 1:size(folder,1) % For each experiment
    movefile(strcat('C:\Users\Marcus\Desktop\Two_Photon\Pre-Process\',... 
        folder(f).name),'C:\Users\Marcus\Desktop\Two_Photon\Processed'); 
end; 

toc

end
     