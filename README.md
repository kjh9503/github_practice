# Personal Information KG-LM Injection System Manual

## 1. Data

- `./data/peacok_person_t1`  
  → Knowledge Graph (KG) data at time point **t1**  
- `./data/peacok_person_t1.2`  
  → KG data at time point **t1.2** (20% of job information modified from t1)

---

## 2. Pretraining and Data Preprocessing

### 2.1 LLM Pretraining (Based on t1)

1. Move to the `./pretraining` directory.  
2. Run the script for the desired model:
   - GPT-J  
     ```
     ./finetune_gptj.sh
     ```
   - Qwen  
     ```
     ./finetune_qwen.sh
     ```
3. The trained model will be saved under `finetuned_models/`.

---

### 2.2 Circuit Discovery (Identifying Personal Information Circuits)

1. Move to the `./circuit_finding` directory.  
2. Extract sample data:
   ```
   ./sample_data.sh  # specify output path using the --out_dir argument
   ```
3. Extract Common Circuits by running one of the following scripts:
   ```
   ./locating_peacok_gptj.sh
   ./locating_peacok_qwen.sh
   ```
   **Arguments:**
   - `--model_dir` : path to the pretrained model from section 2.1  
   - `--sample_dir` : path to the sampled data from step (2)  
4. The circuit results will be saved to the path specified by `--save_path`.  
   - The core file is `info.pt`, which must be moved to `./align/circuits/`.

---

### 2.3 Generating Inputs for the Align Module

#### 2.3.1 KGE Pretraining

1. Move to the `./kge` directory.  
2. Run the following command for both t1 and t1.2:
   ```
   ./run.sh
   ```
3. After training completes, go to the corresponding output path (e.g. `./kge/log/peacok_person_t1_name`) and run:
   ```
   python get_emb.py
   ```
   → Converts KGE results to `.pt` format and saves them in `./align/data/`.

#### 2.3.2 Preparing Align Input Data

1. Move to the `./align` directory.  
2. Generate input data for **updated triples**:
   ```
   python prepare_data.py
   ```
3. Generate input data for **unchanged triples** (for locality testing):
   ```
   python prepare_data_locality.py
   ```

---

## 3. Align Module Training

Move to the `./align` directory.

### 3.1 Training with Common Circuit
```
./run_align_module_common_circuit.sh
```

### 3.2 Training with Personal Circuit
```
./run_align_module_personal_circuit.sh
```

Training logs and results can be found in `./align/log/`.
# spike
# spike
