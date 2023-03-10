# Brain tumor 3D segmentation with AzureML and MONAI (BRATS21)

Glioma brain tumors are among the most aggressive and lethal types of brain tumors. They can cause a range of symptoms, including headaches, seizures, and difficulty with speech and movement. Gliomas can be difficult to diagnose and treat, and early detection is critical for improving patient outcomes.

Computer vision AI has emerged as a promising tool for supporting the diagnosis and treatment of glioma brain tumors. AI algorithms can analyze medical images of the brain and identify the location and extent of tumors with a high degree of accuracy. This can help clinicians make more informed decisions about treatment options, such as surgery or radiation therapy, and monitor the progress of the disease over time. Additionally, AI algorithms can help researchers better understand the underlying biology of gliomas and develop new therapies for this challenging disease.

This demo is based on the [MONAI 3d brain tumor segmentation tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb) and shows how to construct a training workflow of multi-labels segmentation task.

The sub-regions considered for evaluation in the BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by the hyper-intense signal in FLAIR [BraTS 21].

![image](./media/fig_brats21.png)

This notebook has been developed and tested with VSCode connected to an AzureML `STANDARD_D13_V2` Compute Instance using the `azureml_py310_sdkv2` kernel.

## References

[1]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.

[2]: Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[3] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

[4] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q