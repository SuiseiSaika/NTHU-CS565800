# Video Anomaly Detection on CUHK Avenue (Jigsaw-VAD, ECCV 2022)

This project implements and extends **Jigsaw-VAD (ECCV 2022)** for video anomaly detection on the **CUHK Avenue** dataset.  
We experimented with three different temporal prediction strategies and compared their micro-AUROC performance.

---

## 📌 Dataset
- **CUHK Avenue** dataset  
- 16 training videos, 21 testing videos (30,652 frames total)  
- Training set contains *normal* events only  
- Testing set contains *normal + abnormal* events  

---

# 📌 Experiment Settings
According to the course specification:

1. **Spatio-temporal cube:** 5 consecutive frames  
2. **Training videos:** 01–08  
3. **Testing videos:** 01–05  
4. **filter_ratio:** 0.9  
5. **epochs:** 30  

```python
# pkl preprocessing
phase = 'train'
src = f'../avenue/{phase}ing'
detect_dir = f'./detect/avenue_{phase}_detect_result_yolov3.pkl'

with open(detect_dir, 'rb') as f:
    detect = pickle.load(f)
    # Keep 01~08 for train and 01~05 for test as required
with open(detect_dir, 'wb') as pickle_file:
    pickle.dump(detect, pickle_file)
````

---

## 📌 Patch Generation

```bash
python gen_patches.py --phase train --filter_ratio 0.9 --sample_num 5
python gen_patches.py --phase test  --filter_ratio 0.9 --sample_num 5
```

---

## 📌 Training & Testing

```bash
# Training
python main.py --dataset avenue --val_step 100 --print_interval 20 \
    --batch_size 192 --sample_num 5 --epochs 30 --static_threshold 0.2

# Testing
python main.py --dataset avenue --sample_num 5 --checkpoint xxx.pth
```

---

# 🔍 Problem 1 — Original Jigsaw-VAD

**Task:**
Use the default model:

* Spatial jigsaw classification
* Temporal permutation classification

**micro-AUROC:** **0.7944**

---

# 🔍 Problem 2 — Binary Temporal Classification

**Modification:**
Convert temporal permutation prediction into a **binary classifier**:

* **normal:** unpermuted sequence
* **abnormal:** any permuted sequence
* Normal:abnormal ratio = 1:1 during training
* Anomaly score = P(abnormal)

```python
net = model.WideBranchNet(time_length=args.sample_num, num_classes=[1, 81])

binary_labels = (temp_labels[t_flag] == torch.arange(args.sample_num)).all(dim=1)
# balance classes manually
```

**micro-AUROC:** **0.5162**

---

# 🔍 Problem 3 — Full 120-Class Permutation Classification

**Modification:**

* Predict one of **5! = 120** temporal permutations
* Anomaly score = 1 - P(correct permutation)

```python
ps = torch.tensor(list(permutations(range(5)))).cuda()

def gen_v3_label(p):
    index = torch.where((ps == p).all(dim=1))[0].item()
    return F.one_hot(torch.tensor(index), num_classes=120)
```

**micro-AUROC:** **0.6490**

---

# 📊 Final Results Summary

| Version | Description                             | micro-AUROC |
| ------- | --------------------------------------- | ----------- |
| **V1**  | Original Jigsaw-VAD                     | **0.7944**  |
| **V2**  | Binary Classification (Normal/Abnormal) | **0.5162**  |
| **V3**  | 120-Class Permutation Classification    | **0.6490**  |

---

# 🧠 Observations

* **Original Jigsaw-VAD (V1)** achieves the best anomaly detection performance.
* **Binary classification (V2)** oversimplifies the task and loses temporal fine-grained structure.
* **Permutation classification (V3)** is more complex (120 classes), enabling richer temporal modeling but increasing training difficulty.
* Choice of formulation depends on the required **granularity of anomaly scoring**.

---

# 🏁 Conclusion

This project reproduces and extends Jigsaw-VAD with multiple temporal modeling strategies.
Experimental results show that modeling **fine-grained temporal permutations** yields better anomaly detection performance compared to binary simplification, but the original architecture remains the most effective under the given settings.
