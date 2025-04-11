
import wfdb
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import h5py
import os

def load_ecg_record(person_id, record_num):
    record_path = f'ecg-id-database-1.0.0/Person_{person_id:02d}/rec_{record_num}'
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    return record, annotations

def preprocess_ecg(ecg_signal, original_fs, target_fs=250):
    if original_fs != target_fs:
        ecg_signal = signal.resample(ecg_signal, int(len(ecg_signal) * target_fs / original_fs))
    b, a = signal.butter(4, [0.5 / (target_fs / 2), 40 / (target_fs / 2)], btype='bandpass')
    return signal.filtfilt(b, a, ecg_signal)

def segment_heartbeats(ecg_signal, annotations, samples_before=180, samples_after=220):
    r_locations = annotations.sample
    beats = [ecg_signal[r - samples_before: r + samples_after] 
             for r in r_locations 
             if r - samples_before > 0 and r + samples_after < len(ecg_signal)]
    return np.array(beats)

# Specify the persons and record numbers to process
person_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 21, 22, 22, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 30, 30, 30, 30, 30, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 40, 40, 41, 41, 42, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 46, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 57, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 63, 63, 63, 63, 63, 63, 64, 64, 64, 65, 65, 66, 66, 67, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 85]
record_nums = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3]

# Output folder
output_folder = "ecg_results"
os.makedirs(output_folder, exist_ok=True)

for person_id in person_ids:
    for record_num in record_nums:
        try:
            print(f"\n🚀 Processing Person {person_id}, Record {record_num}...")

            # === Load and Preprocess ===
            record, annotations = load_ecg_record(person_id, record_num)
            ecg_signal = record.p_signal[:, 0]
            fs = record.fs
            clean_ecg = preprocess_ecg(ecg_signal, fs)
            beats = segment_heartbeats(clean_ecg, annotations)

            if len(beats) == 0:
                print(f"⚠️ Skipped: No valid beats for Person {person_id}, Record {record_num}")
                continue

            template = np.mean(beats, axis=0)

            # === Plot 1: Raw vs Processed ECG ===
            plt.figure(figsize=(15, 6))
            plt.subplot(2, 1, 1)
            plt.plot(ecg_signal, label='Raw ECG', color='blue', alpha=0.7)
            plt.scatter(annotations.sample, ecg_signal[annotations.sample], color='red', label='R-peaks')
            plt.title("Raw ECG (Before Preprocessing)")
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(clean_ecg, label='Filtered ECG', color='green')
            plt.scatter(annotations.sample, clean_ecg[annotations.sample], color='red', label='R-peaks')
            plt.title("Processed ECG (After Preprocessing)")
            plt.legend()
            plt.grid()

            fig1_path = f"{output_folder}/p{person_id:02d}_r{record_num}_preprocessing.png"
            plt.tight_layout()
            plt.savefig(fig1_path, dpi=300)
            plt.close()

            # === Plot 2: Overlayed Beats ===
            plt.figure(figsize=(15, 6))
            for i, beat in enumerate(beats):
                plt.plot(beat, alpha=0.4, label=f'Beat {i+1}' if i < 5 else None)
            plt.plot(template, color='black', linewidth=3, label='Average Template')
            plt.title(f"Segmented Heartbeats (n={len(beats)})")
            plt.xlabel("Samples from R-peak")
            plt.legend(ncol=5)
            plt.grid()

            fig2_path = f"{output_folder}/p{person_id:02d}_r{record_num}_overlay_beats.png"
            plt.savefig(fig2_path, dpi=300)
            plt.close()

            # === Plot 3: Stacked Beats ===
            plt.figure(figsize=(15, 6))
            for beat in beats:
                plt.plot(beat, alpha=0.3, color='blue')
            plt.plot(template, color='red', linewidth=3, label='Average Template')
            plt.title("Aligned Heartbeats with Template")
            plt.grid()
            plt.legend()

            fig3_path = f"{output_folder}/p{person_id:02d}_r{record_num}_stacked.png"
            plt.savefig(fig3_path, dpi=300)
            plt.close()

            # === Plot 4: Final Template ===
            plt.figure(figsize=(15, 4))
            plt.plot(template, color='red', linewidth=2, label='Average Template')
            plt.title("Final ECG Template")
            plt.grid()
            plt.legend()

            fig4_path = f"{output_folder}/p{person_id:02d}_r{record_num}_template.png"
            plt.savefig(fig4_path, dpi=300)
            plt.close()

            # === HDF5 Save ===
            hdf5_filename = f"{output_folder}/ecg_biometric_p{person_id:02d}_r{record_num}.h5"
            with h5py.File(hdf5_filename, 'w') as f:
                f.create_dataset('clean_ecg', data=clean_ecg)
                f.create_dataset('beats', data=beats)
                f.create_dataset('template', data=template)
                f.create_dataset('r_peaks', data=annotations.sample)
                f.attrs['person_id'] = person_id
                f.attrs['record_num'] = record_num
                f.attrs['sampling_rate'] = fs
                f.attrs['num_beats'] = beats.shape[0]
                f.attrs['samples_per_beat'] = beats.shape[1]

            print(f"✅ Saved plots and HDF5 for Person {person_id}, Record {record_num}.")

        except Exception as e:
            print(f"❌ Error processing Person {person_id}, Record {record_num}: {e}")
