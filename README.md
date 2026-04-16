#License Plate Detection and OCR (MATLAB)
##Overview

This project implements a complete license plate detection and recognition pipeline using MATLAB. The system processes an input vehicle image, identifies the most likely license plate region, and extracts the alphanumeric characters using Optical Character Recognition (OCR).

The goal of this project is to demonstrate algorithm design, image processing techniques, and MATLAB-based system development for real-world applications.

##Features
Adaptive image preprocessing using contrast enhancement (CLAHE)
Gradient-based edge detection for highlighting plate regions
Morphological filtering for noise removal and region consolidation
Candidate region extraction using connected component analysis
Geometric and texture-based scoring of plate candidates
OCR with restricted character set (A–Z, 0–9)
Confidence estimation for detected text
Debug mode with step-by-step visualization of the pipeline
Technologies Used
MATLAB
Image Processing Toolbox
Computer Vision Toolbox
Algorithm Pipeline

##Input Image
→ Grayscale Conversion and Contrast Enhancement
→ Gradient Computation (Edge Detection)
→ Adaptive Thresholding
→ Morphological Operations
→ Connected Components Analysis
→ Candidate Region Scoring and Selection
→ Plate Extraction
→ OCR Processing
→ Text Cleaning and Confidence Estimation

##How It Works
The input image is converted to grayscale and enhanced using adaptive histogram equalization.
Gradient magnitude is computed to highlight strong edges associated with license plates.
Adaptive thresholding and morphological operations are applied to isolate candidate regions.
Connected components are extracted and filtered using geometric constraints such as aspect ratio and area.
Each candidate region is scored based on features like edge density, contrast, and shape.
The highest-scoring region is selected as the license plate.
The selected region is normalized and passed through OCR.
The OCR output is cleaned and evaluated to produce the final plate text and confidence score.
