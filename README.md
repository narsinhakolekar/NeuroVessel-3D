\# NeuroVessel

\## Background-Aware Residual 3D U-Net for Pulmonary Artery and Pulmonary Vein Segmentation from Chest CT



NeuroVessel is a 3D medical image segmentation research project that uses chest CT volumes to segment the \*\*Pulmonary Artery (PA)\*\* and \*\*Pulmonary Vein (PV)\*\*.



The project investigates volumetric deep learning, anatomical segmentation, preprocessing, and evaluation of pulmonary vessels using a three-class segmentation framework.



\*\*Current scope:\*\* PA and PV segmentation. Pulmonary hypertension (PH) risk prediction is not implemented or clinically validated in this version.



\---



\## 1. Research Objectives



The main objectives are:



\- Develop a 3D deep learning model for pulmonary artery and pulmonary vein segmentation.

\- Investigate the use of volumetric CT information for anatomical vessel segmentation.

\- Evaluate segmentation performance using patient-level metrics.

\- Study model development through successive experiments and diagnostic analyses.

\- Provide a visualization interface for inspecting segmentation predictions.



\## 2. Methodology



\### 2.1 Dataset and preprocessing



The project uses the AirRC pulmonary vessel annotation data and corresponding LUNA16 CT data.



The preprocessing pipeline includes:



1\. Reading CT volumes and corresponding annotations.

2\. Resampling CT images to the annotation reference grid using physical-space geometry.

3\. Extracting pulmonary artery annotations for the earlier binary segmentation pipeline.

4\. Preparing multiclass labels for the PA/PV segmentation experiments.

5\. Maintaining patient-level dataset splits to avoid patient overlap between training, validation, and test sets.



The processed CT dataset contains 254 cases. The final evaluation uses a held-out test set of 26 cases.



The V6/V6.1 CT inputs and multiclass labels were audited: 254 CT volumes and 254 labels were matched, with no detected shape, spacing, affine, or orientation mismatches in the audited pairs.



The stored voxel spacing in the audited V6/V6.1 data is 1.0 × 1.0 × 1.0 mm.



\### 2.2 Segmentation classes



The final model predicts three classes:



| Class ID | Label |

|---|---|

| 0 | Background |

| 1 | Pulmonary Artery (PA) |

| 2 | Pulmonary Vein (PV) |



\### 2.3 Model architecture



The final V6.1 model is a \*\*Residual 3D U-Net\*\*.



Main characteristics:



\- 3D volumetric convolutional architecture.

\- Residual feature-learning blocks.

\- Encoder-decoder structure.

\- Skip connections between encoder and decoder.

\- Three output channels for background, PA, and PV.

\- Approximately 1.44 million trainable parameters.



The model was trained using 3D patches of size 96 × 96 × 96.



\## 3. Training and Evaluation Protocol



The final V6.1 checkpoint selected from validation is:



`outputs/checkpoints/airrc\_v6\_1\_multiclass\_best.pt`



The checkpoint corresponds to epoch 29.



The recorded best validation patch Dice was approximately 0.8109. This is a patch-level validation result and should not be interpreted as full-volume test performance.



The final full-volume test protocol uses:



\- 3D sliding-window inference.

\- ROI size: 96 × 96 × 96.

\- Overlap: 0.5.

\- Gaussian blending.

\- Sliding-window batch size: 1.

\- CT intensity clipping: \[-1000, 1000] HU.

\- Intensity scaling to \[0, 1].

\- Three-class argmax prediction.



The final test evaluator is:



`experiments/evaluate\_airrc\_v6\_1\_final\_test.py`



It includes a preflight mode and a separately controlled final-test mode. The held-out test set should remain untouched during model development and parameter selection.



\## 4. Final Test Results



The following are the recorded patient-level macro-average results over 26 held-out test cases.



| Metric | PA | PV |

|---|---:|---:|

| Dice | 78.64% | 71.30% |

| Precision | 74.40% | 64.91% |

| Recall | 83.52% | 79.72% |



\*\*Mean foreground Dice: 74.97%\*\*



These values represent the recorded final evaluation results. The evaluator source code and its test-isolation safeguards were audited; the final test was not independently rerun as part of the repository preparation process.



\### Interpretation



The model achieves higher segmentation performance for the pulmonary artery than for the pulmonary vein in the recorded test evaluation.



The results also indicate that further improvement is needed in vessel boundary accuracy, class separation, and difficult anatomical regions.



\## 5. Repository Structure



```text

NeuroVessel/

│

├── app.py

├── .gitignore

├── requirements.txt

├── prepare\_airrc\_v5.py

├── download\_airrc\_luna16.py

├── verify\_airrc\_v5\_dataset.py

├── visual\_audit\_airrc\_overlays.py

│

├── experiments/

│   ├── prepare\_airrc\_v6\_multiclass.py

│   ├── train\_airrc\_v6\_1.py

│   ├── train\_airrc\_v6\_2.py

│   ├── evaluate\_airrc\_v6\_1\_final\_test.py

│   ├── create\_dataset\_splits.py

│   └── ... supporting and historical experiments

│

├── src/

│   └── filters/

│       └── classical\_filters.py

│

├── data/                 # Local datasets; not tracked by Git

│

└── outputs/               # Local checkpoints and results; not tracked by Git

```



The `experiments/` directory contains the main model-development scripts, evaluation tools, visualization utilities, diagnostic analyses, and historical experiments.



Not every experiment script is part of the final V6.1 inference pipeline.



\## 6. Installation



\### 6.1 Clone the repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd NeuroVessel

```



\### 6.2 Create the Conda environment



```bash

conda create -n neurovessel python=3.11 -y

conda activate neurovessel

```



\### 6.3 Install PyTorch



Install the appropriate PyTorch build for your system.



For the CUDA 12.8 environment used during development:



```bash

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

```



Verify CUDA availability:



```bash

python -c "import torch; print(torch.\_\_version\_\_); print(torch.cuda.is\_available())"

```



\### 6.4 Install remaining dependencies



```bash

pip install -r requirements.txt

```



Note: The PyTorch installation is documented separately because a standard package installation does not guarantee that the CUDA-enabled wheel will be installed.



\## 7. Dataset and Model Artifacts



The `data/` and `outputs/` directories are excluded from Git because they contain large datasets, checkpoints, and generated results.



A public clone of this repository therefore does not automatically contain:



\- Raw CT volumes.

\- Annotation files.

\- Processed CT volumes and labels.

\- Patient split files.

\- Trained model checkpoints.

\- Generated predictions and evaluation outputs.



These artifacts must be obtained or generated separately before running the complete training, evaluation, or demonstration workflow.



The preprocessing scripts may also require adapting local dataset paths. In particular, the V6 multiclass label preparation script currently uses a machine-specific annotation directory.



\## 8. Running the Streamlit Application



The project includes a Streamlit-based 3D segmentation demonstration.



After installing dependencies and placing the required local artifacts in their expected locations, run:



```bash

streamlit run app.py

```



The application uses the V6.1 checkpoint and supports 3D segmentation visualization and NIfTI export.



\*\*Important inference limitation:\*\* The current application does not automatically resample arbitrary uploaded CT volumes to the 1 mm spacing used during model development. Input spacing and preprocessing compatibility must therefore be checked before interpreting predictions. The demonstration should not be considered a clinically validated diagnostic tool.



\## 9. Reproducibility



For reproducibility:



\- Preserve patient-level data splits.

\- Use the documented model checkpoint.

\- Keep the final test set separate from model development.

\- Use the documented sliding-window inference settings.

\- Record the checkpoint and split-manifest hashes when running the final evaluator.

\- Retain per-patient metrics and evaluation outputs.



The final evaluator includes safeguards that require an explicit choice between preflight validation and final-test execution.



\## 10. Limitations



Current limitations include:



1\. Pulmonary vein segmentation remains more challenging than pulmonary artery segmentation.

2\. Segmentation performance can vary across patients and anatomical regions.

3\. The current application does not automatically normalize arbitrary input voxel spacing.

4\. The repository does not include the large datasets or trained checkpoint files.

5\. The model has not been clinically validated.

6\. Pulmonary hypertension risk prediction is outside the implemented scope.

7\. The reported test results are segmentation research metrics and do not establish clinical utility.



\## 11. Future Work



Potential future directions include:



\- Improving PA/PV class separation.

\- Investigating vessel topology and connectivity.

\- Improving robustness to anatomical and acquisition variation.

\- Adding spacing-aware preprocessing to the inference application.

\- Evaluating generalization on independent datasets.

\- Investigating pulmonary hypertension-related measurements as a separate future research direction.



\## 12. Technology Stack



\- Python 3.11

\- PyTorch

\- MONAI

\- SimpleITK

\- NumPy

\- SciPy

\- Scikit-learn

\- Scikit-image

\- NiBabel

\- Streamlit

\- Matplotlib

\- Plotly



\---



\## Research Disclaimer



NeuroVessel is an academic research and development project for 3D pulmonary vessel segmentation. It is not a medical device, is not intended for clinical diagnosis, and has not been clinically validated. Model predictions should not be used as a substitute for professional medical assessment.



\## License



A license has not yet been specified. All rights and dataset usage remain subject to the applicable source dataset terms and permissions.
