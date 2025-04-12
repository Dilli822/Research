
Public ECG datasets (e.g., ECGID, CYBHi).

CYBHi- long-term & short-term
https://zenodo.org/records/2381823#.XT2Ik9VKG2w   

ECGID - 90 PERSON DATASET
https://physionet.org/content/ecgiddb/1.0.0/

https://biosppy.readthedocs.io/en/stable/
https://github.com/Trusted-AI/adversarial-robustness-toolboxpip install adversarial-robustness-toolbox



Experimental Plan (15-Day Version)
Objective
Propose a novel methodology for real-time, privacy-preserving ECG biometrics with spoofing resistance, 
validated with preliminary analysis on existing datasets and simulated wearable constraints.

Scope Adjustment
• ✅ No New Data Collection: Use public ECG datasets (e.g., ECGID, CYBHi).
• Simulation-Based: Simulate wearable conditions (noise, low sampling rate) and edge deployment metrics.
• Preliminary Results: Train a simplified model on a subset of data and extrapolate real-time/spoofing/privacy performance.

1. Materials and Tools
•✅ Datasets: ECGID (85 subjects, PhysioNet), CYBHi (off-person ECG).
•✅ Software: Python, PyTorch, BioSPPy (preprocessing), ART (adversarial attacks), PySyft (federated learning simulation).
•✅ Hardware: Personal computer with GPU (e.g., laptop with NVIDIA GTX 1650 or cloud GPU like Google Colab free tier).

2. Methodology (Simplified)
Step 1: Data Preparation (1 Day)
• ✅ Download ECGID and CYBHi datasets.
• ✅ Preprocess: Bandpass filter (0.5-40 Hz), segment into 5-second windows, normalized to [0, 1].
• ✅ Simulate Wearable Noise: Add Gaussian noise (SNR 20 dB) to mimic wearable artifacts.

Step 2: Model Design (1 Day)
• Architecture: MobileNetV1 (simplified CNN) + GRU, lightweight for edge simulation.
• Input: 5-second ECG segments (500 samples at 100 Hz, downsampled from dataset rates if needed).
• Output: Softmax for identification.

Step 3: Training and Preliminary Results (2 Days)
• Split: 70% train, 15% validation, 15% test (10-20 subjects for speed).
• Baseline: Train on clean ECGID data (10 epochs, Adam, cross-entropy loss).
• Noisy Data: Train on noisy data to simulate wearable conditions.
• Metrics: Report accuracy and EER.

Step 4: Simulated Enhancements (2 Days)
• Spoofing Resistance: Use ART to generate adversarial ECG samples (FGSM), and test detection rate on the baseline model.
• Privacy: Simulate federated learning by training on subject-specific subsets and aggregating weights manually (no raw data sharing).
• Real-Time: Estimate latency/memory using model size and FLOPs (e.g., PyTorch Profiler).

Step 5: Analysis (1 Day)
• Compare baseline vs. noisy accuracy.
• Extrapolate spoofing/privacy performance based on simulation trends.

15-Day Schedule (April 9 - April 24, 2025)
• Day 1-2 (Apr 9-10): Data prep + baseline model training (8-10 hrs).
o Task: Download datasets, preprocess, and train MobileNetV1+GRU on ECGID.
o Output: Baseline accuracy/EER.
• Day 3-4 (Apr 11-12): Simulate wearable noise + spoofing (8-10 hrs).
o Task: Add noise, generate adversarial samples, and test the model.
o Output: Noisy accuracy, spoof detection rate.
• Day 5 (Apr 13): Simulate privacy + real-time (4-6 hrs).
o Task: Federated learning simulation, estimate latency/memory.
o Output: Simulated metrics.
• Day 6 (Apr 14): Outline paper + write Abstract/Introduction (4-6 hrs).
o Task: Draft 650 words.
• Day 7-8 (Apr 15-16): Write Related Work (6-8 hrs).
o Task: Summarize prior papers, and draft 600 words.
• Day 9-10 (Apr 17-18): Write Methodology (8-10 hrs).
o Task: Detail steps, draft 800 words.
• Day 11-12 (Apr 19-20): Write Preliminary Results + Discussion (6-8 hrs).
o Task: Present findings, draft 900 words.
• Day 13 (Apr 21): Write Conclusion + compile References (4-6 hrs).
o Task: Draft 200 words, and list 15-20 citations.
• Day 14 (Apr 22): Revise draft (6-8 hrs).
o Task: Edit for clarity, and coherence, and check word count.
• Day 15 (Apr 23): Final proofread + format (4-6 hrs).
o Task: Polish, submit-ready PDF.
• Apr 24: Buffer day or submission.
Daily Commitment: 4-10 hrs (total ~80-100 hrs).


3. Paper Structure and Writing Schedule
Structure
1. Abstract (100-150 words): Summarize the problem, methodology, and simulated findings.
2. Introduction (500 words): Motivation (ECG biometrics, wearables, privacy/spoofing gaps), objectives.
3. Related Work (600 words): Summarize recent papers (from the prior list), and highlight gap.
4. Methodology (800 words): Detail revised plan (data, model, simulation).
5. Preliminary Results (500 words): Baseline/noisy accuracy, simulated spoofing/privacy metrics.
6. Discussion (400 words): Interpret results, limitations, and future work (full experiment).
7. Conclusion (200 words): Recap contribution, and call for implementation.
8. References: Cite 15-20 sources (prior papers, tools).
Total: ~2,500-3,000 words


4. Expected Preliminary Results
• Baseline: ~90-95% accuracy, EER ~5-10% on ECGID subset.
• Noisy: 5-10% accuracy drop due to wearable simulation.
• Spoofing: ~70-80% detection rate (simulated, unoptimized).
• Privacy: Minimal accuracy loss (<5%) in federated simulation.
• Real-Time: ~50-100 ms latency (estimated on a laptop, adjustable for edge).

5. Tips for Success
• Start Early: Begin data prep and training on Day 1 to avoid delays.
• Use Colab: Free GPU speeds up training if local hardware is slow.
• Leverage Code: Adapt open-source ECG biometric repos (e.g., GitHub) to save time.
• Focus on Proposal: Emphasize methodology novelty; treat results as proof-of-concept.
• Future Work: Propose a complete experiment (from the prior plan) to justify limitations.

Sample Abstract (Draft)
"ECG biometrics offer a unique, secure authentication method, yet their deployment on wearable devices faces real-time processing, privacy, and spoofing vulnerability challenges. This paper proposes a lightweight deep learning model (MobileNetV1+GRU) for ECG-based authentication, integrating federated learning for privacy and adversarial training for spoofing resistance. Using ECGID and CYBHi datasets, we simulate wearable conditions and edge deployment, achieving preliminary accuracy of X% and spoof detection of Y%. While full implementation awaits larger datasets and hardware testing, this methodology bridges critical gaps in scalable, secure biometric systems."

 
 