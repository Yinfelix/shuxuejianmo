function hello_demo()
name = 'math modeling';
values = [1, 2, 3, 4, 5];

fprintf('Hello from Octave in VS Code, %s!\n', name);
fprintf('Mean value: %.2f\n', mean(values));
disp('Squared values:');
disp(values .^ 2);
end