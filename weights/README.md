# Model weights

Each Safetensors file contains a REFINE query head for RET-CLIP visual features and EyeCLIP sentence embeddings. `manifest.json` lists filenames, dimensions, parameter counts and SHA-256 checksums.

Use `refine.load_model(path, device)` to load a head. Obtain the frozen visual and sentence encoders from their authors.

Weights are available for non-commercial research under CC BY-NC-SA 4.0:
https://creativecommons.org/licenses/by-nc-sa/4.0/

Please cite REFINE and its IDRiD/DDR data sources. Third-party encoders retain their own licenses.
