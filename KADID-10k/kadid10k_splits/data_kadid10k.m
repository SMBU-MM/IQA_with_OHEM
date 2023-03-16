% function data_kadid10k(num_selection)
num_selection = 90000;
rng(0);
Dir = './'; %'./kadid10k';
fileID = fopen(fullfile(Dir, 'dmos.csv'));
data = textscan(fileID,'%s %s %f %f\n', 'HeaderLines',1,'Delimiter',',');

imagename = data(:,1);
refnames_all = data(:,2);
mos_original = data(:,3);
std_original = data(:,4);

imagename = imagename{1,1};
refnames_all = refnames_all{1,1};
mos_original = mos_original{1,1};
std_original = std_original{1,1};

mos_original = mos_original';
std_original = sqrt(std_original');

mos = zeros(1,10125,'single');
std = zeros(1,10125,'single');
for i = 1:10125
    mos(1,i) = single(mos_original(i));
    std(1,i) = single(std_original(i));
end
refname = refnames_all(1:125:end);

for split = 1:10
    sel = randperm(81);
    train_path = [];
    train_mos = [];
    train_std = [];
    
    train_path_all = [];
    train_mos_all = [];
    train_std_all = [];
    dist_type = randperm(25);
    % head 
    for i = 1:65
        train_sel = strcmpi(refname(sel(i)),refnames_all );
        train_sel = find(train_sel == 1);
        head_types = dist_type(1:8);
        mid_types = dist_type(9:16);
        tail_types = dist_type(17:25);
        img_org = imagename(train_sel)';
        mos_org = mos_original(train_sel);
        std_org = std_original(train_sel);
        imgs_sel = [];
        mos_sel = [];
        std_sel = [];
        
        imgs_sel_all = [];
        mos_sel_all = [];
        std_sel_all = [];
        for idx = 1:25
            if ismember(idx, head_types) 
                sels_flag = ones(1,5,'single');
                head_sel = find(sels_flag == 1);
                head_imgs = img_org((idx-1)*5+1:idx*5);
                head_moss = mos_org((idx-1)*5+1:idx*5);
                head_stds = std_org((idx-1)*5+1:idx*5);
                imgs_sel = [imgs_sel, head_imgs(head_sel)];
                mos_sel = [mos_sel, head_moss(head_sel)];
                std_sel = [std_sel, head_stds(head_sel)];
                imgs_sel_all = [imgs_sel_all, head_imgs];
                mos_sel_all = [mos_sel_all, head_moss];
                std_sel_all = [std_sel_all, head_stds];
            elseif ismember(idx, mid_types) 
                sels = randperm(5);
                sels_flag = ones(1,5,'single');
                sels_flag(sels(1)) = 0;
                sels_flag(sels(2)) = 0;
                mid_sel = find(sels_flag == 1);
                mid_imgs = img_org((idx-1)*5+1:idx*5);
                mid_moss = mos_org((idx-1)*5+1:idx*5);
                mid_stds = std_org((idx-1)*5+1:idx*5);
                imgs_sel = [imgs_sel, mid_imgs(mid_sel)];
                mos_sel = [mos_sel, mid_moss(mid_sel)];
                std_sel = [std_sel, mid_stds(mid_sel)];
                
                imgs_sel_all = [imgs_sel_all, mid_imgs];
                mos_sel_all = [mos_sel_all, mid_moss];
                std_sel_all = [std_sel_all, mid_stds];
            else
                sels = randperm(5);
                sels_flag = ones(1,5,'single');
                sels_flag(sels(1)) = 0;
                sels_flag(sels(2)) = 0;
                sels_flag(sels(3)) = 0;
                sels_flag(sels(4)) = 0;
                tail_sel = find(sels_flag == 1);
                tail_imgs = img_org((idx-1)*5+1:idx*5);
                tail_moss = mos_org((idx-1)*5+1:idx*5);
                tail_stds = std_org((idx-1)*5+1:idx*5);
                imgs_sel = [imgs_sel, tail_imgs(tail_sel)];
                mos_sel = [mos_sel, tail_moss(tail_sel)];
                std_sel = [std_sel, tail_stds(tail_sel)];  
                
                imgs_sel_all = [imgs_sel_all, tail_imgs];
                mos_sel_all = [mos_sel_all, tail_moss];
                std_sel_all = [std_sel_all, tail_stds];
            end  
        end
        train_path = [train_path, imgs_sel]; 
        train_mos = [train_mos, mos_sel];
        train_std = [train_std, std_sel];   
        
        train_path_all = [train_path_all, imgs_sel_all]; 
        train_mos_all = [train_mos_all, mos_sel_all];
        train_std_all = [train_std_all, std_sel_all];     
     end
    
    test_path = [];
    test_mos = [];
    test_std = [];
    for i = 66:81
        test_sel = strcmpi(refname(sel(i)),refnames_all );
        test_sel = find(test_sel == 1);
        test_path = [test_path, imagename(test_sel)']; 
        test_mos = [test_mos, mos_original(test_sel)];
        test_std = [test_std, std_original(test_sel)];
    end
     
    %% for train split - long tail
    train_index = 1:length(train_mos);
    %all_combination = nchoosek(train_index,2); %全组�?
    all_combination = comb(length(train_index));
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:150*2:end,:);
    
%     fid = fopen(fullfile('./kadid10k/splits2/',num2str(split),'kadid10k_train.txt'),'w');
    fid = fopen(fullfile(num2str(split),'kadid10k_train.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = fullfile('kadid10k/', 'images',train_path(path1_index));
        path1 = strrep(path1,'\','/');
        path1_mos = train_mos(path1_index);
        path1_std = train_std(path1_index);
        path2 = fullfile('kadid10k/', 'images',train_path(path2_index));
        path2 = strrep(path2,'\','/');
        path2_mos = train_mos(path2_index);
        path2_std = train_std(path2_index);
        y = GT_Gaussian(path1_mos, path2_mos, path1_std, path2_std);
        if path1_mos > path2_mos
            yb = 1;
        else
            yb = 0;
        end
        %fprintf(fid,'%s\t%s\t%f\r',path1{1},path2{1},y);
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1{1},path2{1},y, path1_std, path2_std, yb);
    end
    fclose(fid);
    
    %for train_score split#'./kadid10k/splits2',
    fid = fopen(fullfile(num2str(split),'kadid10k_train_score.txt'),'w');
    for i = 1:length(train_path)
        path = fullfile('images',train_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path{1},train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %% for train split- balanced
    train_path = train_path_all;
    train_mos = train_mos_all;
    train_std = train_std_all;  
    
    train_index = 1:length(train_mos);
    %all_combination = nchoosek(train_index,2); %全组�?
    all_combination = comb(length(train_index));
    num_combines = size(all_combination);
    selected_index = randperm(num_combines(1));
    selected_index = selected_index(1:num_selection);
    combination = all_combination(selected_index,:);
    %combination = all_combination(selected_index,:);
    %combination = combination(1:150*2:end,:);
    
%     fid = fopen(fullfile('./kadid10k/splits2/',num2str(split),'kadid10k_train.txt'),'w');
    fid = fopen(fullfile(num2str(split),'kadid10k_train_balanced.txt'),'w');
    for i = 1:length(combination)
        path1_index = combination(i,1);
        path2_index = combination(i,2);
        path1 = fullfile('kadid10k/', 'images',train_path(path1_index));
        path1 = strrep(path1,'\','/');
        path1_mos = train_mos(path1_index);
        path1_std = train_std(path1_index);
        path2 = fullfile('kadid10k/', 'images',train_path(path2_index));
        path2 = strrep(path2,'\','/');
        path2_mos = train_mos(path2_index);
        path2_std = train_std(path2_index);
        y = GT_Gaussian(path1_mos, path2_mos, path1_std, path2_std);
        if path1_mos > path2_mos
            yb = 1;
        else
            yb = 0;
        end
        %fprintf(fid,'%s\t%s\t%f\r',path1{1},path2{1},y);
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1{1},path2{1},y, path1_std, path2_std, yb);
    end
    fclose(fid);
    
    %for train_score split#'./kadid10k/splits2',
    fid = fopen(fullfile(num2str(split),'kadid10k_train_score_balanced.txt'),'w');
    for i = 1:length(train_path)
        path = fullfile('images',train_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path{1},train_mos(i),train_std(i));
    end
    fclose(fid);
    
    %% for test split #'./kadid10k/splits2',
    fid = fopen(fullfile(num2str(split),'kadid10k_test.txt'),'w');
    head_fid = fopen(fullfile(num2str(split),'kadid10k_test_head.txt'),'w');
    mid_fid = fopen(fullfile(num2str(split),'kadid10k_test_mid.txt'),'w');
    tail_fid = fopen(fullfile(num2str(split),'kadid10k_test_tail.txt'),'w');
    for i = 1:length(test_path)
        path = fullfile('images',test_path(i));
        path = strrep(path,'\','/');
        fprintf(fid,'%s\t%f\t%f\r',path{1},test_mos(i),test_std(i));
        strs = strsplit(path{1}, '_');
        type_idx = str2num(strs{2});
        type_idx
        if ismember(type_idx, head_types)
            fprintf(head_fid,'%s\t%f\t%f\r',path{1},test_mos(i),test_std(i));
        elseif ismember(type_idx, mid_types)
            fprintf(mid_fid,'%s\t%f\t%f\r',path{1},test_mos(i),test_std(i));
        else
            fprintf(tail_fid,'%s\t%f\t%f\r',path{1},test_mos(i),test_std(i));
        end
    end
    fclose(fid);  
end

disp('kadid10k completed!');
