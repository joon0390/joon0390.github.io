---
title: "Molecular Toxicity Prediction and Principled Uncertainty Quantification using a Domain-Informed Composite Kernel Gaussian Process Classifier"
excerpt: "A study that performs toxicity classification and uncertainty estimation using a composite kernel incorporating molecular structure information and domain knowledge."
date: 2026-05-21
collection: projects
layout: single
order: 6
classes:
  - wide
tags:
  - Gaussian Process
  - Kernel Methods
  - Molecular Toxicity
  - Uncertainty Quantification
  - Bayesian ML
---

## Project Summary

- **Introduction:** A study combining SMILES-based molecular structures and chemical domain features into a composite kernel to jointly predict toxicity probability and uncertainty.
- **Period:** 2026.04-
- **Data:** SMILES, Morgan fingerprints, RDKit descriptors, PCA-based continuous features, toxicity labels
- **Tech Stack:** RDKit, Gaussian Process Classifier, Tanimoto/RBF/Matern Kernel, Composite Kernel, Calibration Analysis
- **Achievement (Performance):** Achieved the most balanced performance compared to baseline models, recording `F1 0.803`, `AUC 0.877`, and `Brier 0.147` based on the Hybrid GP.

## Problem Definition

Molecular toxicity prediction does not end with simply guessing whether a molecule is toxic or not. In actual candidate screening, it is crucial to observe how confident the model is, whether its predictions are unstable for molecules structurally distant from the training data, and whether there are any judgments that contradict chemical intuition.

Instead of relying on a single black-box classifier to predict toxicity labels, this project combined molecular structural similarity and domain-based continuous features at the kernel level. This was designed by connecting the composite kernel to a Gaussian Process Classifier to provide not only the probability of toxicity but also the prediction uncertainty.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/arch.png" alt="Domain-informed molecular toxicity prediction composite kernel architecture">
  <figcaption>The overall architecture: extracting Morgan fingerprints and RDKit descriptors from SMILES, synthesizing structural similarity and continuous feature kernels, and inputting them into the Gaussian Process Classifier.</figcaption>
</figure>

## Data and EDA

The data utilized SMILES representations of molecules and toxicity labels as the base units, incorporating both structure-based binary fingerprints and continuous descriptors. In the EDA, we focused on checking the distribution of toxic/non-toxic classes, structural similarity based on Morgan fingerprints, the scale and correlation structure of RDKit descriptors, and separability in the continuous feature space after PCA.

Particularly in toxicity prediction, it is dangerous if the model is overly confident about molecules that are structurally far from the training data. Therefore, the model's performance was evaluated not just by F1 or AUC, but by jointly observing the Brier score and the calibration trends across different kernel combinations.

## Approach

The core approach is not to rely on a single general-purpose kernel, but to synthesize similarities from different perspectives. The Tanimoto kernel reflects structural similarity based on Morgan fingerprints, while the RBF/Matern kernel reflects the distance structure of RDKit descriptors and the PCA-based continuous feature space.

1. Extract Morgan fingerprints from SMILES to represent the substructure similarity of the molecular structure.
2. Calculate RDKit descriptors and compress them using PCA to form a continuous, property-based feature space.
3. Combine Tanimoto, RBF, and Matern kernels to simultaneously reflect structural similarity and domain features.
4. Connect the composite kernel to a Gaussian Process Classifier to jointly estimate the toxicity probability and posterior uncertainty.

<figure class="project-figure project-figure--medium">
  <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/kernels.png" alt="Comparison of F1, Brier, and AUC by composite kernel combination for molecular toxicity prediction">
  <figcaption>As a result of comparing kernel combinations, the Tanimoto + RBF + Matern combination showed the best balance across F1, AUC, and Brier scores.</figcaption>
</figure>

## Achievements (Performance)

- The proposed Hybrid GP recorded `F1 0.803`, `AUC 0.877`, and `Brier 0.147`.
- The single Tanimoto kernel-based GP showed levels of `F1 0.763`, `AUC 0.863`, and `Brier 0.165`, while the RBF single kernel was relatively lower at `F1 0.709`, `AUC 0.798`, and `Brier 0.231`.
- Compared to traditional classifiers like LDA, QDA, and Naive Bayes, the Hybrid GP demonstrated the highest performance based on the F1 score.
- The key to this performance improvement was handling both fingerprint structural similarity and descriptor-based continuous features together within a single probabilistic kernel model.

<div class="project-figure-grid">
  <figure class="project-figure">
    <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/AUC.png" alt="AUC comparison by kernel combination">
    <figcaption>The ranking performance for toxicity classification was generally high in Tanimoto-based combinations, and the model combining all three kernels recorded the highest AUC.</figcaption>
  </figure>
  <figure class="project-figure">
    <img src="/assets/img/projects/domain-informed-molecular-toxicity-gpc/f1_model.png" alt="F1 performance comparison by model">
    <figcaption>The Hybrid GP outperformed both traditional classifiers and single-kernel GPs, achieving the best balance in toxic/non-toxic classification.</figcaption>
  </figure>
</div>

## Lessons Learned

The biggest takeaway from this project was that in molecular toxicity prediction, `from what perspective we consider molecules to be similar` is far more important than the choice of the model itself. Initially, I was more interested in the Gaussian Process Classifier as a model, but as experiments progressed, the performance differences largely depended on how the kernel was designed and which molecular representations were combined.

In particular, while the Tanimoto kernel captured the structural similarity of molecular fingerprints well, there was clearly continuous property information that could not be explained by it alone. Therefore, it was necessary to handle RDKit descriptors and PCA-based features together using RBF and Matern kernels, and this combination produced better results than a single kernel.

Another lesson learned is that in problems like toxicity prediction, which can directly link to real-world decision-making, high accuracy alone is not sufficient. It was essential to look at which samples the model was confident about, which ones it was uncertain about, and how well those probabilities were calibrated. Through this process, I realized once again the importance of evaluating models by looking at performance metrics, calibration, and uncertainty interpretation all together.

## Papers / Resources

- Original PDF: [Open Domain-informed Molecular Toxicity GPC PDF](/assets/papers/domain-informed-molecular-toxicity-gpc.pdf)