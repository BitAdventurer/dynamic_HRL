import torch
import os

# 모델 클래스 import (HRL.py에 정의된 클래스와 동일해야 함)
from HRL import MacroQNet, MicroQNet, CRNNClassifier
from config import get_config

from models import TemporalEmbedding, ROIEmbedding, MacroQNet, MicroQNet, CRNNClassifier

# 환경/하이퍼파라미터 세팅 (HRL.py와 동일)
args = get_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = args.feature_dim
MACRO_IN_DIM = args.macro_in_dim
MICRO_IN_DIM = args.micro_in_dim
RL_HIDDEN_DIM = args.rl_hidden_dim
RL_EMBED_DIM = args.rl_embed_dim
CRNN_OUT_CHANNELS = args.crnn_out_channels
CRNN_KERNEL_SIZE = args.crnn_kernel_size
CRNN_POOL_SIZE = args.crnn_pool_size
CRNN_HIDDEN_DIM = args.crnn_hidden_dim
CRNN_NUM_LAYERS = args.crnn_num_layers
CRNN_DROPOUT = args.dropout_rate
CRNN_NUM_CLASSES = args.crnn_num_classes

possible_window_ratios = [i / 100 for i in range(1, 101)]
possible_shift_ratios = [i / 100 for i in range(1, 101)]

# 변환 대상 파일 목록
model_dir = os.path.join(os.path.dirname(__file__), '../models/weight')
file_map = []
for i in range(1, 6):
    file_map.append((f"macroQ_fold{i}.pt", MacroQNet, (MACRO_IN_DIM, RL_HIDDEN_DIM, RL_EMBED_DIM, len(possible_window_ratios))))
    file_map.append((f"microQ_fold{i}.pt", MicroQNet, (MICRO_IN_DIM, RL_HIDDEN_DIM, RL_EMBED_DIM, len(possible_shift_ratios))))
    file_map.append((f"classifier_fold{i}.pt", CRNNClassifier, (FEATURE_DIM, CRNN_HIDDEN_DIM, CRNN_NUM_LAYERS, CRNN_DROPOUT, CRNN_NUM_CLASSES, CRNN_OUT_CHANNELS, CRNN_KERNEL_SIZE, CRNN_POOL_SIZE)))

for fname, cls, ctor_args in file_map:
    fpath = os.path.join(model_dir, fname)
    if not os.path.exists(fpath):
        print(f"[SKIP] {fpath} (not found)")
        continue
    print(f"[CONVERT] {fpath}")
    model = torch.load(fpath, map_location=DEVICE)
    # 새 state_dict 파일명
    state_dict_path = fpath.replace('.pt', '_state.pt')
    # 만약 이미 state_dict 파일이면 skip
    if os.path.exists(state_dict_path):
        print(f"[SKIP] {state_dict_path} (already exists)")
        continue
    # state_dict 저장
    torch.save(model.state_dict(), state_dict_path)
    print(f"[SAVED] {state_dict_path}")

print("변환 완료.")
