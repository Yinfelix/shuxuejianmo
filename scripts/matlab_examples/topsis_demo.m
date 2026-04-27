function result = topsis_demo()
% Simple TOPSIS demo for contest-style multi-criteria evaluation.

data = [
    85  7.2  120;
    78  8.1  150;
    92  6.8  110;
    88  7.5  130
];

% 1 means benefit criterion, -1 means cost criterion.
criterion_type = [1, 1, -1];
weights = [0.4, 0.3, 0.3];

normalized = zeros(size(data));
for column_index = 1:size(data, 2)
    column = data(:, column_index);
    if criterion_type(column_index) == 1
        transformed = column;
    else
        transformed = min(column) ./ column;
    end
    normalized(:, column_index) = transformed ./ sqrt(sum(transformed .^ 2));
end

weighted = normalized .* weights;
positive_ideal = max(weighted, [], 1);
negative_ideal = min(weighted, [], 1);

distance_positive = sqrt(sum((weighted - positive_ideal) .^ 2, 2));
distance_negative = sqrt(sum((weighted - negative_ideal) .^ 2, 2));
score = distance_negative ./ (distance_positive + distance_negative);

[ranking_score, ranking_order] = sort(score, 'descend');

result.weighted_matrix = weighted;
result.score = score;
result.ranking_score = ranking_score;
result.ranking_order = ranking_order;

disp('topis score:');
disp(score);
disp('Ranking order:');
disp(ranking_order);
end