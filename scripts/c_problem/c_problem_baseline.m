script_dir = fileparts(mfilename('fullpath'));
workspace_root = fileparts(fileparts(script_dir));

summary_path = fullfile(workspace_root, 'outputs', 'tables', 'c_problem_baseline_summary.csv');
detail_path = fullfile(workspace_root, 'data', 'processed', 'c_problem_node_detail.csv');

if ~exist(summary_path, 'file')
    error('Missing baseline summary CSV: %s. Run the Python baseline script first.', summary_path);
end

if ~exist(detail_path, 'file')
    error('Missing node detail CSV: %s. Run the Python baseline script first.', detail_path);
end

summary_text = fileread(summary_path);
detail_text = fileread(detail_path);
detail_lines = strsplit(detail_text, {'\r\n', '\n', '\r'});
detail_lines = detail_lines(~cellfun('isempty', detail_lines));

fprintf('C problem baseline summary (from CSV)\n');
fprintf('===================================\n');
fprintf('%s\n', summary_text);

fprintf('Top node detail preview\n');
fprintf('=======================\n');
preview_count = min(6, numel(detail_lines));
for i = 1:preview_count
    fprintf('%s\n', detail_lines{i});
end
