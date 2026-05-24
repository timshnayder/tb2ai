import os
import json
import sqlite3
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

#device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_DIR = PROJECT_ROOT / "modeltraining"
VOCAB_PATH_SPRAY = AI_DIR / "vocab_spray.json"
VOCAB_COORDS_PATH_SPRAY = AI_DIR / "vocab_coords_spray.json"
VOCAB_PATH_MIRROR = AI_DIR / "vocab_mirror.json"
VOCAB_COORDS_PATH_MIRROR = AI_DIR / "vocab_coords_mirror.json"
SPRAY_MODEL_PATH = AI_DIR / "conditional_climb_ai_spray.pth"
MIRROR_MODEL_PATH = AI_DIR / "conditional_climb_ai_mirror.pth"
METADATA_PATH = PROJECT_ROOT / "data" / "tension" / "board_metadata.json"

# Load metadata once on startup
board_metadata = {}
if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r") as f:
            board_metadata = json.load(f)
    except Exception as e:
        logging.error(f"Error loading board metadata JSON: {str(e)}")

# Global states
vocab_spray = None
reverse_vocab_spray = None
coord_tensor_spray = None

vocab_mirror = None
reverse_vocab_mirror = None
coord_tensor_mirror = None

spray_model = None
mirror_model = None

GRADE_MAP = {
    '3C': 9, '4A': 10, '4B': 11, '4C': 12, 'V0': 12,
    '5A': 13, 'V1': 14, 
    '5B': 14, '5C': 15, 'V2': 15,
    '6A': 16, '6A+': 17, 'V3': 16, 
    '6B': 18, '6B+': 19, 'V4': 18,
    '6C': 20, '6C+': 21, 'V5': 20,
    '7A': 22, 'V6': 22, 
    '7A+': 23, 'V7': 23,
    '7B': 24, '7B+': 25, 'V8': 24,
    '7C': 26, 'V9': 26, 
    '7C+': 27, 'V10': 27,
    '8A': 28, 'V11': 28, 
    '8A+': 29, 'V12': 29,
    '8B': 30, 'V13': 30, 
    '8B+': 31, 'V14': 31,
    '8C': 32, 'V15': 32, 
    '8C+': 33, 'V16': 33,
    '9A': 34, 'V17': 34, 
    '9A+': 35, 'V18': 35,
    '9B': 36, 'V19': 36, 
    '9B+': 37, 'V20': 37,
    '9C': 38, 'V21': 38, 
    '9C+': 39, 'V22': 39
}

class ConditionalClimbGenerator(nn.Module):
    def __init__(self, vocab_size, num_grades, max_angle, embedding_dim, hidden_dim, coord_tensor):
        super(ConditionalClimbGenerator, self).__init__()
        self.hold_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.grade_embedding = nn.Embedding(num_grades, embedding_dim)
        #half size so doesn't overpower holds
        self.angle_embedding = nn.Embedding(max_angle, embedding_dim//2)
        
        self.register_buffer('coords', coord_tensor)
        # hold (64) + grade (64) + angle(32) + coord x(1) + coord y(1) + no_match(1)
        input_size = (embedding_dim*2) + (embedding_dim//2)+3
        
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, holds, grades, angles, no_match):
        hold_embeds = self.hold_embedding(holds)
        grade_embeds = self.grade_embedding(grades).unsqueeze(1).repeat(1, holds.size(1), 1)
        #embed the angle and expand it
        angle_embeds = self.angle_embedding(angles).unsqueeze(1).repeat(1, holds.size(1), 1)
        
        no_match_expanded = no_match.unsqueeze(1).unsqueeze(2).repeat(1, holds.size(1), 1)
        hold_coords = self.coords[holds]

        combined_input = torch.cat((hold_embeds, grade_embeds, angle_embeds, hold_coords, no_match_expanded), dim=2)
        lstm_out, _ = self.lstm(combined_input)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# Initialize models and load trained brain weights
def init_generator():
    """Load vocab, coordinates, Spray model, and Mirror model dynamically at startup."""
    global vocab_spray, reverse_vocab_spray, coord_tensor_spray
    global vocab_mirror, reverse_vocab_mirror, coord_tensor_mirror
    global spray_model, mirror_model
    
    logger.info("Initializing user's AI climb generator...")
    try:
        NUM_GRADES = 45
        
        # 1. Load Spray Layout Vocabulary and Model
        logger.info(f"Loading Spray vocab from {VOCAB_PATH_SPRAY}...")
        with open(VOCAB_PATH_SPRAY, 'r') as f:
            vocab_spray = json.load(f)
        reverse_vocab_spray = {v: k for k, v in vocab_spray.items()}
        
        with open(VOCAB_COORDS_PATH_SPRAY, 'r') as f:
            vocab_coords_dict_spray = json.load(f)
            
        coord_matrix_spray = [vocab_coords_dict_spray[str(i)] for i in range(len(vocab_spray))]
        coord_tensor_spray = torch.tensor(coord_matrix_spray, dtype=torch.float32).to(device)
        
        SPRAY_VOCAB_SIZE = len(vocab_spray)
        logger.info(f"Loading Spray model from {SPRAY_MODEL_PATH} (vocab size: {SPRAY_VOCAB_SIZE})...")
        spray_model = ConditionalClimbGenerator(SPRAY_VOCAB_SIZE, NUM_GRADES, max_angle=15, embedding_dim=64, hidden_dim=256, coord_tensor=coord_tensor_spray)
        spray_model.load_state_dict(torch.load(SPRAY_MODEL_PATH, map_location=device))
        spray_model = spray_model.to(device)
        spray_model.eval()
        
        # 2. Load Mirror Layout Vocabulary and Model
        logger.info(f"Loading Mirror vocab from {VOCAB_PATH_MIRROR}...")
        with open(VOCAB_PATH_MIRROR, 'r') as f:
            vocab_mirror = json.load(f)
        reverse_vocab_mirror = {v: k for k, v in vocab_mirror.items()}
        
        with open(VOCAB_COORDS_PATH_MIRROR, 'r') as f:
            vocab_coords_dict_mirror = json.load(f)
            
        coord_matrix_mirror = [vocab_coords_dict_mirror[str(i)] for i in range(len(vocab_mirror))]
        coord_tensor_mirror = torch.tensor(coord_matrix_mirror, dtype=torch.float32).to(device)
        
        MIRROR_VOCAB_SIZE = len(vocab_mirror)
        logger.info(f"Loading Mirror model from {MIRROR_MODEL_PATH} (vocab size: {MIRROR_VOCAB_SIZE})...")
        mirror_model = ConditionalClimbGenerator(MIRROR_VOCAB_SIZE, NUM_GRADES, max_angle=15, embedding_dim=64, hidden_dim=256, coord_tensor=coord_tensor_mirror)
        mirror_model.load_state_dict(torch.load(MIRROR_MODEL_PATH, map_location=device))
        mirror_model = mirror_model.to(device)
        mirror_model.eval()
        
        logger.info("User's AI generator initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing user's AI generator: {str(e)}", exc_info=True)
        raise e

# =======================================================
# 3. GENERATION FUNCTION WITH GUARDRAILS (USER'S EXACT LOGIC)
# =======================================================
def generate_new_climb(inputGrade, temperature, maxLen=20, target_layout_id=11, size_id=8, angle=40, is_nomatch=False, beam_width=4):
    # Select active vocabulary, reverse vocabulary, and model based on target_layout_id (10 is Mirror, 11 is Spray)
    if target_layout_id == 10:
        active_vocab = vocab_mirror
        active_reverse_vocab = reverse_vocab_mirror
        active_model = mirror_model
    else:
        active_vocab = vocab_spray
        active_reverse_vocab = reverse_vocab_spray
        active_model = spray_model

    gradeKey = inputGrade.strip().upper()
    if gradeKey not in GRADE_MAP:
        raise KeyError(f"Grade '{inputGrade}' is unrecognized. Use formats like 'V5', '6C', '7A+', etc.")
    
    gradeValue = GRADE_MAP[gradeKey]
    # Convert input angle (degrees) to angle index (increments of 5)
    angle_val = int(angle) // 5
    # Clamp safely within the embedding vocabulary [0, 14]
    angle_val = max(0, min(14, angle_val))
    no_match_val = 1.0 if is_nomatch else 0.0

    # Load valid placements from JSON metadata
    valid_placements = set()
    try:
        valid_placements = set(board_metadata.get("layouts", {}).get(str(target_layout_id), {}).get("sizes", {}).get(str(size_id), {}).get("valid_placements", []))
        logger.info(f"Loaded {len(valid_placements)} valid placements from JSON metadata for layout {target_layout_id}, size {size_id}")
    except Exception as e:
        logger.error(f"Error loading valid placements from JSON metadata: {str(e)}")

    print(f"\nGenerating a {inputGrade} climb with Beam Search (Width: {beam_width}, Angle: {angle}°, No-Match: {is_nomatch}) for board size {size_id}...")
    print("-" * 55)

    # Initialize beam search candidates starting with [START]
    candidates = [{
        "sequence": [active_vocab["[START]"]],
        "log_prob": 0.0,
        "start_holds_placed": 0,
        "finish_holds_placed": 0
    }]
    
    completed_climbs = []

    for step in range(maxLen):
        if not candidates:
            break
            
        holds_batch = []
        for cand in candidates:
            seq = cand["sequence"]
            pad_len = 46 - len(seq)
            padded_seq = ([0] * pad_len if pad_len > 0 else []) + seq
            holds_batch.append(padded_seq)

        input_holds = torch.tensor(holds_batch, dtype=torch.long).to(device)
        input_grade = torch.tensor([gradeValue] * len(candidates), dtype=torch.long).to(device)
        input_angle = torch.tensor([angle_val] * len(candidates), dtype=torch.long).to(device)
        input_no_match = torch.tensor([no_match_val] * len(candidates), dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = active_model(input_holds, input_grade, input_angle, input_no_match)

        logits = logits / temperature
        log_probs_batch = torch.log_softmax(logits, dim=-1).cpu().numpy()

        all_expansions = []
        for c_idx, cand in enumerate(candidates):
            log_probs = log_probs_batch[c_idx].copy()
            start_holds_placed = cand["start_holds_placed"]
            finish_holds_placed = cand["finish_holds_placed"]

            # Apply guardrails
            for i in range(len(log_probs)):
                token_str = active_reverse_vocab.get(i, "")
                #pad or start
                if i == 0 or i == 1:
                    log_probs[i] = -float('inf')
                    continue
                
                # Guardrail: Limit choices to holds that fit inside the selected board size
                if valid_placements and "r" in token_str and "p" in token_str:
                    try:
                        placement_str = token_str.split('r')[0].replace('p', '')
                        placement_id = int(placement_str)
                        if placement_id not in valid_placements:
                            log_probs[i] = -float('inf')
                            continue
                    except ValueError:
                        pass

                # max 2 start holds allowed 
                if "r5" in token_str and start_holds_placed >= 2:
                    log_probs[i] = -float('inf')
                    continue

                # do not allow the climb to end if no start holds or no finish holds are placed yet
                if i == active_vocab["[END]"]:
                    if start_holds_placed == 0 or finish_holds_placed == 0:
                        log_probs[i] = -float('inf')
                        continue

                # if we are running out of steps and have no finish holds, force a finish hold
                if finish_holds_placed == 0 and step >= maxLen - 2:
                    if not ("r7" in token_str):
                        log_probs[i] = -float('inf')
                        continue

                # if we have 1 finish hold, the next move MUST be the [END] token, OR a second finish hold.
                if finish_holds_placed == 1:
                    
                    if not (i == active_vocab["[END]"] or "r7" in token_str):
                        log_probs[i] = -float('inf')
                        continue
                # if we already have 2 finish holds, the climb is over.
                if finish_holds_placed >= 2:
                    
                    if i != active_vocab["[END]"]:
                        log_probs[i] = -float('inf')
                        continue

            # To introduce diversity in bouldering options (so clicking Generate doesn't return the identical climb every time),
            # we can add a tiny random noise to log_probs, OR select expansions using a mixture of probability and randomness.
            # Adding a tiny amount of Gumbel noise to log_probs is a mathematically standard way to do stochastic beam search!
            # Gumbel noise = -log(-log(uniform(0, 1)))
            # Scale of noise = 0.1 * temperature (smaller noise means more deterministic, larger means more creative)
            if temperature > 0:
                gumbel_noise = -np.log(-np.log(np.random.uniform(1e-10, 1.0, size=len(log_probs))))
                log_probs_adjusted = log_probs + temperature * gumbel_noise
            else:
                log_probs_adjusted = log_probs

            # now time for beam search 
            # get the top K indices for this candidate
            top_k_indices = np.argsort(log_probs_adjusted)[-beam_width:]
            
            for idx in top_k_indices:
                val = log_probs[idx]
                if val == -float('inf'):
                    continue
                
                next_token_str = active_reverse_vocab.get(idx, "")
                next_start = start_holds_placed + (1 if "r5" in next_token_str else 0)
                next_finish = finish_holds_placed + (1 if "r7" in next_token_str else 0)
                
                new_cand = {
                    "sequence": cand["sequence"] + [idx],
                    "log_prob": cand["log_prob"] + val,
                    "start_holds_placed": next_start,
                    "finish_holds_placed": next_finish
                }
                all_expansions.append(new_cand)

        if not all_expansions:
            break

        # Sort all expansions by their accumulated log probability
        all_expansions = sorted(all_expansions, key=lambda x: x["log_prob"], reverse=True)
        
        # Select the top beam_width expansions
        next_candidates = []
        for cand in all_expansions:
            if len(next_candidates) >= beam_width:
                break
                
            last_token = cand["sequence"][-1]
            if last_token == active_vocab["[END]"]:
                completed_climbs.append(cand)
            else:
                next_candidates.append(cand)
                
        candidates = next_candidates

    # Choose the best climb from completed ones, or fallback to the best active candidate
    best_climb = None
    if completed_climbs:
        completed_climbs = sorted(completed_climbs, key=lambda x: x["log_prob"], reverse=True)
        best_climb = completed_climbs[0]["sequence"]
        print(f"-> Selected best completed climb with log-prob: {completed_climbs[0]['log_prob']:.4f}")
    elif candidates:
        best_climb = candidates[0]["sequence"]
        print(f"-> Fallback: Selected best active climb with log-prob: {candidates[0]['log_prob']:.4f}")
    else:
        best_climb = [active_vocab["[START]"]]
        print("-> Warning: Generation bottlenecked completely.")

    # Remove [START] and [END] from output string
    generated_tokens_list = []
    role_names = {
        '5': 'Start (Green)',
        '6': 'Hand/Foot (Blue)',
        '7': 'Finish (Red)',
        '8': 'Foot-Only (Orange)'
    }
    
    for token_id in best_climb:
        if token_id in [active_vocab["[START]"], active_vocab["[END]"]]:
            continue
        raw_string = active_reverse_vocab[token_id]
        generated_tokens_list.append(raw_string)
        
        hold_num = raw_string.split('r')[0].replace('p', '')
        role_num = raw_string.split('r')[1]
        print(f"Hold ID: {hold_num:4} | Color role: {role_names.get(role_num, f'Role {role_num}')}")

    print("-" * 55)
    visualizer_ready_string = "".join(generated_tokens_list)
    print("COPY/PASTE THIS STRING INTO YOUR VISUALIZER SCRIPT:")
    print(f'ai_generated_string = "{visualizer_ready_string}"\n')

    return {
        "grade": inputGrade,
        "raw_frames": visualizer_ready_string,
        "frames": visualizer_ready_string,
        "layout_id": target_layout_id,
        "size_id": size_id,
        "angle": angle,
        "is_nomatch": is_nomatch
    }

# Flask integration wrapper function
def generate_climb_for_grade(grade_name, temperature=0.7, target_layout_id=11, size_id=8, max_len=20, angle=40, is_nomatch=False, beam_width=4):
    return generate_new_climb(
        inputGrade=grade_name,
        temperature=temperature,
        maxLen=max_len,
        target_layout_id=target_layout_id,
        size_id=size_id,
        angle=angle,
        is_nomatch=is_nomatch,
        beam_width=beam_width
    )
