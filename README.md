# VirMODE---RNA-Virus-Transmission

Predicting virus-host associations and transmission modes is essential for understanding viral ecology, host range determination, and the potential for cross-species transmission events. The rapid expansion of viral genomic and proteomic data, coupled with advances in structural bioinformatics, has created unprecedented opportunities to decode the molecular determinants of viral transmission.
ViralMODE addresses this challenge by providing an accessible, scalable, and biologically interpretable AI-fueled pipeline for large-scale viral protein analysis and transmission prediction. The pipeline introduces a conceptual structural framework for exploring the modular architecture of viral proteins through a Modulome-driven approach. It integrates geometric motif detection, low-dimensional structural embeddings, intrinsic disorder modeling, and sequence-derived energetic profiling to decode protein biophysical determinants of viral host range and transmission.
ViralMODE's Modulome quantifies a rich suite of biophysical properties—including solvent accessibility, charge distribution, protonation states, conformational flexibility, energetic signatures, and residue-level spatial networks. The pipeline employs protein embeddings that capture stronger predictive signals than traditional sequence composition features.
By training an ensemble of AI models (Random Forest, XGBoost, LightGBM, CatBoost) on these Modulome-derived biophysical signatures, ViralMODE demonstrates variable yet robust performance across multiple prediction tasks. ViralMODE incorporates explainable AI through SHAP analysis, identifying important proteins and structural features that significantly contribute to transmission predictions, revealing that protein domain architectures—particularly PF, G3DSA, and SSF structural modules—along with proteome disorder properties, serve as the primary determinants of viral transmission modes and host specificity.
In summary, ViralMODE provides a powerful, flexible, and user-friendly platform that merges AI-fueled ensemble learning with Modulome-driven structural bioinformatics to expand our understanding of viral transmission mechanisms.


What Makes This Special
Handles class imbalance: Applied scale_pos_weight, balanced class weights
Robust metrics: Used MCC instead of accuracy (better for imbalanced data)
Explainable AI: SHAP values show why models make predictions
Multiple algorithms: Tested 4 different ensemble methods per target
Complete reproducibility: Random seed 42, all hyperparameters documented
Dashboard-ready: Pre-formatted JSON, CSV, high-quality images
