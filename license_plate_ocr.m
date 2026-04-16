function results = license_plate_ocr(imageInput, debugMode)
% LICENSE_PLATE_OCR Detect a license plate and read characters using OCR.
%
% Usage:
%   results = license_plate_ocr("car.jpg", true);
%   results = license_plate_ocr(imread("car.jpg"), false);
%
% Inputs:
%   imageInput - image file path, or RGB/grayscale image matrix
%   debugMode  - true/false to show intermediate processing steps
%
% Outputs:
%   results - struct with fields:
%       .plateText
%       .boundingBox
%       .confidence
%       .allCandidates
%
% Notes:
%   This version is built to look polished for a portfolio project:
%   - adaptive preprocessing
%   - candidate region scoring
%   - morphology + geometry filtering
%   - OCR with restricted character set
%   - confidence estimation
%
% Requires:
%   Image Processing Toolbox
%   Computer Vision Toolbox (for ocr)

    if nargin < 2
        debugMode = false;
    end

    I = loadInputImage(imageInput);
    Iorig = I;

    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end

    Igray = im2uint8(Igray);

    % -------- Preprocessing --------
    Ieq = adapthisteq(Igray, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    Ifilt = medfilt2(Ieq, [3 3]);

    % Gradient emphasizes high-contrast plate characters and borders
    Gx = imfilter(double(Ifilt), [-1 0 1; -2 0 2; -1 0 1], 'replicate');
    Gy = imfilter(double(Ifilt), [-1 -2 -1; 0 0 0; 1 2 1], 'replicate');
    Gmag = mat2gray(sqrt(Gx.^2 + Gy.^2));

    % Morphological consolidation for plate-like rectangular regions
    bw = imbinarize(Gmag, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.45);
    bw = imclose(bw, strel('rectangle', [5 17]));
    bw = imopen(bw, strel('rectangle', [3 5]));
    bw = imfill(bw, 'holes');
    bw = bwareaopen(bw, 150);

    % Connected component analysis
    cc = bwconncomp(bw);
    stats = regionprops(cc, 'BoundingBox', 'Area', 'Extent', 'Eccentricity', 'Image');

    candidates = [];
    plateBoxes = [];
    scores = [];

    imgArea = size(Igray,1) * size(Igray,2);

    for k = 1:numel(stats)
        bb = stats(k).BoundingBox;
        w = bb(3);
        h = bb(4);
        aspectRatio = w / max(h,1);
        areaRatio = stats(k).Area / imgArea;
        extentVal = stats(k).Extent;
        ecc = stats(k).Eccentricity;

        % Plate heuristics: wide rectangle, medium density, not too small
        validAR = aspectRatio >= 2.0 && aspectRatio <= 7.5;
        validArea = areaRatio >= 0.002 && areaRatio <= 0.20;
        validExtent = extentVal >= 0.35 && extentVal <= 0.95;
        validEcc = ecc >= 0.75;

        if validAR && validArea && validExtent && validEcc
            roi = imcrop(Igray, bb);
            roiScore = scorePlateCandidate(roi, aspectRatio, areaRatio, extentVal);

            candidates = [candidates; struct( ...
                'BoundingBox', bb, ...
                'Score', roiScore, ...
                'AspectRatio', aspectRatio, ...
                'AreaRatio', areaRatio, ...
                'Extent', extentVal)]; %#ok<AGROW>

            plateBoxes = [plateBoxes; bb]; %#ok<AGROW>
            scores = [scores; roiScore]; %#ok<AGROW>
        end
    end

    if isempty(candidates)
        results = struct( ...
            'plateText', "", ...
            'boundingBox', [], ...
            'confidence', 0, ...
            'allCandidates', []);
        warning('No likely license plate region was found.');
        return;
    end

    [~, bestIdx] = max(scores);
    bestBox = candidates(bestIdx).BoundingBox;
    plateROI = imcrop(Iorig, bestBox);

    % -------- Plate normalization for OCR --------
    plateGray = ensureGray(plateROI);
    plateGray = imresize(plateGray, [160 NaN]);
    plateGray = im2uint8(plateGray);
    plateGray = adapthisteq(plateGray);
    plateGray = imsharpen(plateGray, 'Radius', 1.2, 'Amount', 1.3);

    % Generate a few OCR-ready variants and choose the best
    variants = buildOCRVariants(plateGray);

    bestText = "";
    bestConfidence = -inf;

    for i = 1:numel(variants)
        ocrResult = ocr(variants{i}, ...
            'TextLayout', 'Block', ...
            'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789');

        cleaned = cleanPlateText(ocrResult.Text);
        conf = estimateConfidence(ocrResult.WordConfidences, cleaned);

        if strlength(cleaned) >= 4 && conf > bestConfidence
            bestText = cleaned;
            bestConfidence = conf;
        end
    end

    if bestConfidence < 0
        bestConfidence = 0;
    end

    results = struct( ...
        'plateText', bestText, ...
        'boundingBox', bestBox, ...
        'confidence', bestConfidence, ...
        'allCandidates', candidates);

    if debugMode
        showDebugViews(Iorig, Igray, Ieq, Gmag, bw, bestBox, plateGray, bestText, bestConfidence);
    end
end

% ========================= Helper Functions =========================

function I = loadInputImage(imageInput)
    if ischar(imageInput) || isstring(imageInput)
        I = imread(imageInput);
    elseif isnumeric(imageInput)
        I = imageInput;
    else
        error('imageInput must be a file path or an image matrix.');
    end
end

function G = ensureGray(I)
    if size(I,3) == 3
        G = rgb2gray(I);
    else
        G = I;
    end
end

function score = scorePlateCandidate(roi, aspectRatio, areaRatio, extentVal)
    roi = ensureGray(roi);
    roi = im2uint8(roi);
    edgeDensity = nnz(edge(roi, 'Canny')) / numel(roi);
    contrastVal = std(double(roi(:))) / 255;

    % Weighted score
    score = 0.35 * min(aspectRatio / 4.5, 1.5) + ...
            0.20 * min(edgeDensity * 8, 1.5) + ...
            0.25 * min(contrastVal * 4, 1.5) + ...
            0.10 * min(extentVal / 0.7, 1.5) + ...
            0.10 * min(areaRatio * 100, 1.5);
end

function variants = buildOCRVariants(I)
    bw1 = imbinarize(I, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.42);
    bw2 = imbinarize(I, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.55);
    bw3 = ~imbinarize(I, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.48);

    bw1 = postProcessTextMask(bw1);
    bw2 = postProcessTextMask(bw2);
    bw3 = postProcessTextMask(bw3);

    variants = {I, bw1, bw2, bw3};
end

function bw = postProcessTextMask(bw)
    bw = bwareaopen(bw, 20);
    bw = imopen(bw, strel('rectangle', [2 2]));
    bw = imclose(bw, strel('rectangle', [2 3]));
end

function txt = cleanPlateText(rawText)
    txt = upper(string(rawText));
    txt = regexprep(txt, '[^A-Z0-9]', '');
    txt = strtrim(txt);

    % Optional trimming for implausibly long OCR output
    if strlength(txt) > 10
        txt = extractBetween(txt, 1, 10);
        txt = string(txt);
    end
end

function conf = estimateConfidence(wordConfidences, textVal)
    if isempty(textVal) || strlength(textVal) == 0
        conf = 0;
        return;
    end

    wc = wordConfidences(~isnan(wordConfidences));
    if isempty(wc)
        conf = 35;
        return;
    end

    conf = mean(wc);

    % Mild bonus for plausible license plate lengths
    len = strlength(textVal);
    if len >= 5 && len <= 8
        conf = conf + 5;
    end

    conf = min(conf, 100);
end

function showDebugViews(Iorig, Igray, Ieq, Gmag, bw, bestBox, plateGray, plateText, conf)
    figure('Name','License Plate OCR Debug','NumberTitle','off');

    subplot(2,4,1); imshow(Iorig); title('Original');
    subplot(2,4,2); imshow(Igray); title('Grayscale');
    subplot(2,4,3); imshow(Ieq); title('CLAHE Enhanced');
    subplot(2,4,4); imshow(Gmag); title('Gradient Magnitude');
    subplot(2,4,5); imshow(bw); title('Candidate Mask');

    subplot(2,4,6); imshow(Iorig); title('Best Plate Box');
    hold on;
    rectangle('Position', bestBox, 'EdgeColor', 'g', 'LineWidth', 2);
    hold off;

    subplot(2,4,7); imshow(plateGray); title('Plate ROI for OCR');

    subplot(2,4,8); axis off;
    text(0.05, 0.7, "Detected Text: " + plateText, 'FontSize', 12, 'Interpreter', 'none');
    text(0.05, 0.5, sprintf('Confidence: %.2f', conf), 'FontSize', 12);
end
