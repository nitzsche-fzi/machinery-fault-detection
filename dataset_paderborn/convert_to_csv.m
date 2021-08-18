%
% This is the code for converting the '.mat' files of the
% Paderborn dataset to '.csv' files that are usable by the
% Python code.
%

input_path = "data_paderborn";
dataset_dir_names = get_directories(input_path);

for i = 1 : length(dataset_dir_names)
    % Create path for directory
    path_name = fullfile(input_path, dataset_dir_names(i));
    disp(path_name);
    % Find all files in the directory
    bearing_files = get_directories(path_name);
    for j = 1 : length(bearing_files)
        % Check if file is .mat file
        if contains(bearing_files(j), ".mat")
            % Create full name
            full_name = fullfile(path_name, bearing_files(j));
            % Load and access data
            mat_data = load(full_name);
            mat_data_cell = struct2cell(mat_data);
            raw_data = mat_data_cell{1}.Y(7).Data.';
            % Create output path
            output_path = strrep(full_name, ".mat", ".csv");
            % Write .csv file
            csvwrite(output_path, raw_data);
        end
    end
end

function dir_names = get_directories(path)
    dirs = dir(path);
    dir_names = string({dirs.name}.');
    dir_names = dir_names(3:end);
end
