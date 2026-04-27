function interpolation_demo()
output_dir = fullfile('..', '..', 'outputs', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

x = [0, 1, 2, 3, 4, 5, 6];
y = [0, 0.8, 0.9, 0.1, -0.8, -1.0, -0.2];
xq = linspace(min(x), max(x), 400);

y_linear = interp1(x, y, xq, 'linear');
y_spline = interp1(x, y, xq, 'spline');
y_pchip = interp1(x, y, xq, 'pchip');

figure_handle = figure('visible', 'off');
graphics_toolkit(figure_handle, 'gnuplot');
plot(x, y, 'ko', 'markerfacecolor', 'k', 'markersize', 6);
hold on;
plot(xq, y_linear, 'b-', 'linewidth', 1.5);
plot(xq, y_spline, 'r-', 'linewidth', 1.5);
plot(xq, y_pchip, 'g-', 'linewidth', 1.5);
grid on;
legend('Sample points', 'Linear', 'Spline', 'PCHIP', 'location', 'northeast');
title('Interpolation Curve Demo');
xlabel('x');
ylabel('y');

output_file = fullfile(output_dir, 'interpolation_demo.png');
print(figure_handle, output_file, '-dpng', '-r150');
close;

fprintf('Interpolation figure saved to:\n%s\n', output_file);
end