clc; clear; close all;

% Replace with your own test image path
imagePath = 'sample_car.jpg';

results = license_plate_ocr(imagePath, true);

fprintf('Detected Plate Text: %s\n', string(results.plateText));
fprintf('Confidence: %.2f\n', results.confidence);

I = imread(imagePath);
figure('Name', 'Final Detection', 'NumberTitle', 'off');
imshow(I); hold on;
if ~isempty(results.boundingBox)
    rectangle('Position', results.boundingBox, 'EdgeColor', 'y', 'LineWidth', 3);
    label = sprintf('Plate: %s | Conf: %.1f', string(results.plateText), results.confidence);
    text(results.boundingBox(1), max(results.boundingBox(2)-10, 10), label, ...
        'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'black');
end
hold off;
