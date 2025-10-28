import logging
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset

from .models import CRNNClassifier, MacroQNet, MicroQNet, SpatioTemporalTransformer
from utils.preprocessing import preprocess_matrix

class DFCSequenceDataset(Dataset):
    def __init__(self, dfc_arr: np.ndarray, label_arr: np.ndarray, subject_ids: np.ndarray, 
                 device_if_gpu: Optional[torch.device] = None, use_pca: bool = True, pca_dim: int = 32,
                 threshold: float = 0.01, pca_model=None, use_lda: bool = False, lda_dim: int = 1, lda_model=None):
        super().__init__()
        self.pca_model = pca_model
        self.lda_model = lda_model

        if dfc_arr.ndim == 4 and dfc_arr.shape[2] == 116 and dfc_arr.shape[3] == 116:
            N, T, _, _ = dfc_arr.shape
            temp = np.zeros((N, T, 6670), dtype=np.float32)
            for i in range(N):
                for t in range(T):
                    temp[i, t] = preprocess_matrix(dfc_arr[i, t], threshold=threshold)
            temp_2d = temp.reshape(-1, 6670)
            if use_lda:
                labels_2d = np.repeat(label_arr, T)
                n_classes = len(np.unique(labels_2d))
                lda_dim_effective = min(lda_dim, n_classes - 1)
                if self.lda_model is None:
                    self.lda_model = LinearDiscriminantAnalysis(n_components=lda_dim_effective)
                    temp_2d = self.lda_model.fit_transform(temp_2d, labels_2d)
                else:
                    temp_2d = self.lda_model.transform(temp_2d)
                temp = temp_2d.reshape(N, T, lda_dim_effective)
            elif use_pca:
                if self.pca_model is None:
                    self.pca_model = PCA(n_components=pca_dim)
                    temp_2d = self.pca_model.fit_transform(temp_2d)
                else:
                    temp_2d = self.pca_model.transform(temp_2d)
                temp = temp_2d.reshape(N, T, pca_dim)
            self.dfc_arr = temp
        elif dfc_arr.ndim == 3 and dfc_arr.shape[-1] == 116:
            # Apply PCA/LDA to [N, T, 116] format as well
            N, T, F = dfc_arr.shape
            temp_2d = dfc_arr.reshape(-1, F)
            if use_lda:
                labels_2d = np.repeat(label_arr, T)
                n_classes = len(np.unique(labels_2d))
                lda_dim_effective = min(lda_dim, n_classes - 1)
                if self.lda_model is None:
                    self.lda_model = LinearDiscriminantAnalysis(n_components=lda_dim_effective)
                    temp_2d = self.lda_model.fit_transform(temp_2d, labels_2d)
                else:
                    temp_2d = self.lda_model.transform(temp_2d)
                temp = temp_2d.reshape(N, T, lda_dim_effective)
            elif use_pca:
                if self.pca_model is None:
                    self.pca_model = PCA(n_components=pca_dim)
                    temp_2d = self.pca_model.fit_transform(temp_2d)
                else:
                    temp_2d = self.pca_model.transform(temp_2d)
                temp = temp_2d.reshape(N, T, pca_dim)
            else:
                temp = dfc_arr
            self.dfc_arr = temp
        else:
            self.dfc_arr = dfc_arr

        self.label_arr = label_arr
        self.subject_ids = subject_ids
        self.use_gpu_data = (device_if_gpu is not None)

        if self.use_gpu_data:
            self.dfc_gpu = torch.tensor(self.dfc_arr, dtype=torch.float32, device=device_if_gpu)
            self.label_gpu = torch.tensor(label_arr, dtype=torch.long, device=device_if_gpu)
        else:
            self.dfc_gpu = None
            self.label_gpu = None

    def __len__(self) -> int:
        return len(self.dfc_arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Any]:
        if self.use_gpu_data:
            seq_data = self.dfc_gpu[idx]
            label = self.label_gpu[idx].item()
        else:
            seq_data = torch.tensor(self.dfc_arr[idx], dtype=torch.float32)
            label = self.label_arr[idx]
        subj_id = self.subject_ids[idx]
        return seq_data, label, subj_id

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple[Any, ...]] = []
        self.idx = 0

    def push(self, transition: Tuple[Any, ...]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int, seed: int = None) -> List[Tuple[Any, ...]]:
        rng = random.Random(seed) if seed is not None else random
        return rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

class HRL(nn.Module):
    def __init__(self, feature_dim, crnn_hidden_dim, crnn_num_layers, crnn_dropout, crnn_num_classes,
                 crnn_out_channels, crnn_kernel_size, crnn_pool_size,
                 macro_in_dim, micro_in_dim, rl_hidden_dim, rl_embed_dim,
                 num_macro_actions, num_micro_actions, rl_dropout, crnn_bidirectional=False,
                 classifier_type='crnn', transformer_num_heads=8, transformer_dim_feedforward=512, transformer_dropout=0.1):
        super(HRL, self).__init__()
        
        self.classifier_type = classifier_type
        
        # Create classifier based on type
        if classifier_type == 'transformer':
            self.classifier = SpatioTemporalTransformer(
                input_dim=feature_dim,
                hidden_dim=crnn_hidden_dim,
                num_layers=crnn_num_layers,
                num_heads=transformer_num_heads,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                num_classes=crnn_num_classes
            )
        elif classifier_type == 'cbgru':
            self.classifier = CRNNClassifier(
                input_dim=feature_dim,
                hidden_dim=crnn_hidden_dim,
                num_layers=crnn_num_layers,
                dropout=crnn_dropout,
                num_classes=crnn_num_classes,
                out_channels=crnn_out_channels,
                kernel_size=crnn_kernel_size,
                pool_size=crnn_pool_size,
                bidirectional=True
            )
        else:  # 'crnn' (default)
            self.classifier = CRNNClassifier(
                input_dim=feature_dim,
                hidden_dim=crnn_hidden_dim,
                num_layers=crnn_num_layers,
                dropout=crnn_dropout,
                num_classes=crnn_num_classes,
                out_channels=crnn_out_channels,
                kernel_size=crnn_kernel_size,
                pool_size=crnn_pool_size,
                bidirectional=crnn_bidirectional
            )
        self.macroQ = MacroQNet(macro_in_dim, rl_hidden_dim, rl_embed_dim, num_macro_actions, dropout=rl_dropout)
        self.microQ = MicroQNet(micro_in_dim, rl_hidden_dim, rl_embed_dim, num_micro_actions, dropout=rl_dropout)

    def train(self, mode=True):
        self.classifier.train(mode)
        self.macroQ.train(mode)
        self.microQ.train(mode)
        return self

    def eval(self):
        self.classifier.eval()
        self.macroQ.eval()
        self.microQ.eval()
        return self

    def evaluate_and_get_rewards(self, loader, criterion, device, ablation_mode,
                                 fixed_window_size, fixed_shift_ratio,
                                 macro_in_dim, micro_in_dim,
                                 possible_window_sizes, possible_shift_ratios,
                                 max_steps_per_trial, fold_idx=0,
                                 dataset_type='validation', track_usage=False, verbose=True, save_dir=None):
        # Wrapper for compatibility with train.py
        return evaluate_dataset(
            (self.classifier, self.macroQ, self.microQ),
            loader, criterion, device, ablation_mode,
            fixed_window_size, fixed_shift_ratio,
            macro_in_dim, micro_in_dim,
            possible_window_sizes, possible_shift_ratios,
            max_steps_per_trial, fold_idx, dataset_type, verbose, track_usage
        )

    def state_dict(self):
        return {
            'classifier': self.classifier.state_dict(),
            'macroQ': self.macroQ.state_dict(),
            'microQ': self.microQ.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.classifier.load_state_dict(state_dict['classifier'])
        self.macroQ.load_state_dict(state_dict.get('macroQ', {}))
        self.microQ.load_state_dict(state_dict.get('microQ', {}))

def update_metrics(pred: int, lbl: int, metrics: Dict[str, int]):
    if pred == 1 and lbl == 1: metrics['TP'] += 1
    elif pred == 1 and lbl == 0: metrics['FP'] += 1
    elif pred == 0 and lbl == 1: metrics['FN'] += 1
    else: metrics['TN'] += 1

def calc_scores(metrics: Dict[str, int]) -> Tuple[float, float, float, float]:
    TP, TN, FP, FN = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
    eps = 1e-8
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    sen = TP / (TP + FN + eps)
    spec = TN / (TN + FP + eps)
    f1 = (2 * TP) / (2 * TP + FP + FN + eps)
    return acc, sen, spec, f1

def compute_macro_dqn_loss(batch, macro_qnet, macro_target1, macro_target2, gamma, device):
    """
    Macro-agent DQN learning (standard DQN with bootstrapping)
    """
    states = torch.cat([b[0] for b in batch], dim=0).to(device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float, device=device).view(-1, 1)
    next_states = torch.cat([b[3] for b in batch], dim=0).to(device)
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float, device=device).view(-1, 1)

    q_sa = macro_qnet(states).gather(1, actions)
    with torch.no_grad():
        q_next1 = macro_target1(next_states)
        q_next2 = macro_target2(next_states)
        q_next_min = torch.min(q_next1, q_next2)
        q_next_max = q_next_min.max(dim=1, keepdim=True)[0]
        target = rewards + gamma * (1 - dones) * q_next_max
    return F.mse_loss(q_sa, target)

def compute_micro_bandit_loss(batch, micro_qnet, device):
    """
    Micro-agent contextual bandit learning (Paper-aligned)
    Target: y_t^micro = r_t (immediate reward only, no gamma)
    """
    states = torch.cat([b[0] for b in batch], dim=0).to(device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float, device=device).view(-1, 1)
    # No next_states or dones needed for bandit
    
    q_sa = micro_qnet(states).gather(1, actions)
    target = rewards  # Paper: immediate reward only, no bootstrapping
    return F.mse_loss(q_sa, target)

def evaluate_dataset(model_tuple, loader, criterion, device, ablation_mode, fixed_window_size, fixed_shift_ratio,
                     macro_in_dim, micro_in_dim, possible_window_sizes, possible_shift_ratios, max_steps_per_trial,
                     fold_idx=0, dataset_type='validation', verbose=True, track_usage=False):
    classifier, macroQ, microQ = model_tuple
    classifier.eval()
    macroQ.eval()
    microQ.eval()

    patient_predictions = defaultdict(list)
    patient_labels = {}
    patient_probs = defaultdict(list)
    tmetric = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    fold_window_usage_mdd = defaultdict(lambda: defaultdict(Counter))
    fold_window_usage_nc = defaultdict(lambda: defaultdict(Counter))
    fold_step_usage_mdd = defaultdict(lambda: defaultdict(Counter))
    fold_step_usage_nc = defaultdict(lambda: defaultdict(Counter))
    fold_reward_mdd = defaultdict(lambda: defaultdict(list))
    fold_reward_nc = defaultdict(lambda: defaultdict(list))
    
    # Collect RL transitions for replay buffer
    macro_transitions = []
    micro_transitions = []

    with torch.no_grad():
        for dfc_batch, lb_batch, pid_batch in loader:
            dfc_batch = dfc_batch.to(device)
            for i in range(dfc_batch.size(0)):
                dfc_seq_i = dfc_batch[i]
                lb_i = lb_batch[i].item()
                pid_i = pid_batch[i]

                current_idx = 0
                step_count = 0
                done = False

                while not done:
                    remain_len = dfc_seq_i.size(0) - current_idx
                    if remain_len <= 0:
                        break

                    macro_s_test = dfc_seq_i[current_idx:].mean(dim=0, keepdim=True)
                    if macro_s_test.size(1) < macro_in_dim:
                        macro_s_test = F.pad(macro_s_test, (0, macro_in_dim - macro_s_test.size(1)), value=0)
                    micro_s_test = torch.zeros(1, micro_in_dim, device=device)

                    if ablation_mode == 'full':
                        macro_qvals = macroQ(macro_s_test)
                        macro_a_test = macro_qvals.argmax(dim=1).item()
                        w = possible_window_sizes[macro_a_test]  # Paper: absolute size
                        w = min(w, remain_len)  # Don't exceed remaining length
                        w = max(w, 1)  # Minimum 1
                        
                        micro_qvals = microQ(micro_s_test)
                        micro_a_test = micro_qvals.argmax(dim=1).item()
                        shift_ratio = possible_shift_ratios[micro_a_test]  # ρ_t
                        # Paper: δ_t = max{1, round(ρ_t × w_t)}
                        delta_t = max(1, round(shift_ratio * w))

                    elif ablation_mode == 'no_macro':
                        w = min(fixed_window_size, remain_len)
                        w = max(w, 1)  # 최소 1 보장
                        macro_a_test = 0  # 로깅용 값 (의미 없음)
                        
                        # Micro 에이전트는 정상 작동
                        micro_qvals = microQ(micro_s_test)
                        micro_a_test= micro_qvals.argmax(dim=1).item()
                        shift_ratio = possible_shift_ratios[micro_a_test]
                        delta_t = max(1, round(shift_ratio * w))
                    
                    elif ablation_mode == 'no_micro':
                        macro_qvals = macroQ(macro_s_test)
                        macro_a_test = macro_qvals.argmax(dim=1).item()
                        w = possible_window_sizes[macro_a_test]  # Paper: absolute size
                        w = min(w, remain_len)
                        w = max(w, 1)  # 최소 1
                        
                        # Micro 에이전트 비활성화: 고정된 이동 간격 사용
                        shift_ratio = fixed_shift_ratio
                        delta_t = max(1, round(shift_ratio * w))
                        micro_a_test = 0  # 로깅용 값 (의미 없음)
                    
                    else:  # 'fixed'
                        # 두 에이전트 모두 비활성화: 고정 값 사용
                        w = min(fixed_window_size, remain_len)
                        w = max(w, 1)  # 최소 1
                        shift_ratio = fixed_shift_ratio
                        delta_t = max(1, round(shift_ratio * w))
                        macro_a_test = 0  # 로깅용 값 (의미 없음)
                        micro_a_test= 0  # 로깅용 값 (의미 없음)

                    seq_segment_test = dfc_seq_i[current_idx:current_idx + w].unsqueeze(0)  # (1, w, feat_dim)
                    out_test = classifier(seq_segment_test)
                    pred = out_test.argmax(dim=1).item()

                    # Calculate reward for RL (always calculate, regardless of track_usage)
                    reward = 0.0  # Default
                    if ablation_mode == 'full':
                        # reward 계산 (정확한 예측은 양수 reward, 잘못된 예측은 음수 reward)
                        prob = F.softmax(out_test, dim=1)
                        confidence = prob[0, pred].item()  # Probability of the predicted class

                        # --- 1. Base Reward Calculation (Paper-aligned) ---
                        if pred == lb_i:  # Correct prediction
                            if lb_i == 0:  # True Negative (Correctly predicted NC)
                                base_reward = 0.5
                            else:  # True Positive (Correctly predicted MDD)
                                base_reward = 0.5
                        else:  # Incorrect prediction
                            if lb_i == 0 and pred == 1:  # False Positive (NC misclassified as MDD)
                                base_reward = -1.0
                            elif lb_i == 1 and pred == 0:  # False Negative (MDD misclassified as NC)
                                base_reward = -0.5  # Paper: -0.5 (changed from -0.3)
                            else: # Should not happen if pred and lb_i are 0 or 1
                                base_reward = -0.5 # Default penalty for unexpected cases

                        # --- 2. Confidence Scaling (Paper formula) ---
                        # Paper: r_t = b_t × (0.75 + 0.5 × p_max)
                        confidence_factor = 0.75 + 0.5 * confidence
                        reward = base_reward * confidence_factor
                        # Removed: diversity bonus, baseline subtraction, clipping
                    else:
                        # For non-full modes, set reward to 0 (no RL)
                        reward = 0.0
                    
                    # Track usage statistics (optional, only if track_usage is enabled)
                    if track_usage and ablation_mode == 'full':
                        if lb_i == 1:  # MDD
                            fold_window_usage_mdd[fold_idx][pid_i][w] += 1
                            fold_step_usage_mdd[fold_idx][pid_i][shift_ratio] += 1
                            fold_reward_mdd[fold_idx][pid_i].append((w, shift_ratio, reward))
                        else:  # NC
                            fold_window_usage_nc[fold_idx][pid_i][w] += 1
                            fold_step_usage_nc[fold_idx][pid_i][shift_ratio] += 1
                            fold_reward_nc[fold_idx][pid_i].append((w, shift_ratio, reward))

                    prob_1 = F.softmax(out_test, dim=1)[:,1]
                    patient_probs[pid_i].append(prob_1.item())
                    patient_predictions[pid_i].append(pred)
                    patient_labels[pid_i] = lb_i

                    # micro_s_new
                    micro_s_new_test = seq_segment_test.view(1, -1)
                    if micro_s_new_test.shape[1] < micro_in_dim:
                            micro_s_new_test = F.pad(micro_s_new_test, (0, micro_in_dim - micro_s_new_test.shape[1]), value=0)
                    elif micro_s_new_test.shape[1] > micro_in_dim:
                            micro_s_new_test = micro_s_new_test[:, :micro_in_dim]

                    # Paper: use delta_t for step size (already calculated above)
                    current_idx += delta_t
                    step_count  += 1

                    if step_count>=max_steps_per_trial or current_idx>=dfc_seq_i.size(0):
                        done=True
                    else:
                        remain_len = dfc_seq_i.size(0) - current_idx
                        if remain_len <= 0:
                            done=True
                        else:
                            macro_s_next_test = dfc_seq_i[current_idx:].mean(dim=0, keepdim=True)
                            if macro_s_next_test.size(1) < macro_in_dim:
                                    macro_s_next_test = F.pad(macro_s_next_test, (0, macro_in_dim - macro_s_next_test.size(1)), value=0)
                            
                            # Store transitions for RL training (train phase only)
                            if ablation_mode == 'full':
                                # Macro transition: (state, action, reward, next_state, done)
                                macro_transitions.append((
                                    macro_s_test.cpu(),
                                    macro_a_test,
                                    reward,
                                    macro_s_next_test.cpu(),
                                    0.0  # not done
                                ))
                                # Micro transition: (state, action, reward) - no next_state for bandit
                                micro_transitions.append((
                                    micro_s_test.cpu(),
                                    micro_a_test,
                                    reward,
                                    micro_s_new_test.cpu(),  # dummy next_state
                                    0.0  # dummy done
                                ))
                            
                            macro_s_test = macro_s_next_test
                            micro_s_test = micro_s_new_test

    for pid, preds in patient_predictions.items():
        maj_vote = 1 if sum(preds) >= (len(preds) / 2) else 0
        update_metrics(maj_vote, patient_labels[pid], tmetric)

    acc_val, sen_val, spec_val, f1_val = calc_scores(tmetric)
    
    # ROC AUC calculation
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    for pid, label in patient_labels.items():
        if pid in patient_probs:
            y_true_list.append(label)
            # 예측값은 patient_predictions의 다수결로 계산
            preds = patient_predictions[pid]
            maj_vote = 1 if sum(preds) >= (len(preds) / 2) else 0
            y_pred_list.append(maj_vote)
            y_score_list.append(np.mean(patient_probs[pid]))

    # 진단용: 예측/라벨 분포, confusion matrix 출력
    from collections import Counter
    try:
        import sklearn.metrics
        print("[EVAL_DIAG] y_true 분포:", Counter(y_true_list))
        print("[EVAL_DIAG] y_pred 분포:", Counter(y_pred_list))
        print("[EVAL_DIAG] Confusion Matrix:\n", sklearn.metrics.confusion_matrix(y_true_list, y_pred_list))
    except Exception as e:
        print("[EVAL_DIAG] Confusion matrix 계산 오류:", e)

    if len(np.unique(y_true_list)) > 1:
        fpr, tpr, _ = roc_curve(y_true_list, y_score_list, pos_label=1)
        roc_auc_val = auc(fpr, tpr)
    else:
        roc_auc_val = 0.5 # Or handle as per desired logic

    import logging
    logger = logging.getLogger("evaluate_dataset")
    logger.info(f"[EVAL_METRIC_LOG] acc={acc_val:.4f}, sen={sen_val:.4f}, spec={spec_val:.4f}, f1={f1_val:.4f}, auc={roc_auc_val:.4f}")
    logger.info(f"[EVAL_METRIC_LOG] y_true_list={y_true_list}")
    logger.info(f"[EVAL_METRIC_LOG] y_score_list={y_score_list}")
    # --- 평균 loss 계산 및 결과에 추가 ---
    # 모든 예측에 대해 loss 누적
    total_loss = 0.0
    total_count = 0
    classifier.eval()
    with torch.no_grad():
        for dfc_batch, lb_batch, pid_batch in loader:
            dfc_batch = dfc_batch.to(device)
            lb_batch = lb_batch.to(device)
            out = classifier(dfc_batch)
            loss = criterion(out, lb_batch)
            loss_scalar = loss.mean()  # 또는 loss.sum() / batch_size
            total_loss += loss_scalar.item() * dfc_batch.size(0)
            total_count += dfc_batch.size(0)
    avg_loss = total_loss / max(total_count, 1)

    results = {
        'acc': acc_val, 'sen': sen_val, 'spec': spec_val, 'f1': f1_val, 'auc': roc_auc_val,
        'loss': avg_loss,
        'patient_labels': patient_labels, 'patient_predictions': patient_predictions, 'patient_probs': patient_probs,
        'macro_transitions': macro_transitions,  # For RL training
        'micro_transitions': micro_transitions   # For RL training
    }

    if track_usage and ablation_mode == 'full':
        results.update({
            'window_usage_mdd': {fold_idx: dict(fold_window_usage_mdd[fold_idx])},
            'window_usage_nc': {fold_idx: dict(fold_window_usage_nc[fold_idx])},
            'step_usage_mdd': {fold_idx: dict(fold_step_usage_mdd[fold_idx])},
            'step_usage_nc': {fold_idx: dict(fold_step_usage_nc[fold_idx])},
            'reward_mdd': {fold_idx: dict(fold_reward_mdd[fold_idx])},
            'reward_nc': {fold_idx: dict(fold_reward_nc[fold_idx])}
        })
    return results