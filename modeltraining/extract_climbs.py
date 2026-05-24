import sqlite3
import re
import json
# original = 9
# mirror = 10
# spray = 11

#12 x 8
#edge_top <= 140
#edge_left >=-40
#edge_right <=40
#edge_bottom >=4

#r1 start?
#r2 hand?
#r3 end?
#r4 foot?

#r5 start
#r6 hold
#r7 end
#r8 foot

# Layout configuration:
# 10 = mirror
# 11 = spray
LAYOUT_ID = 11

connection = sqlite3.connect('tension_db.db')
cursor = connection.cursor()

# We join the climbs table with climb_stats to get the consensus difficulty

query = """
    SELECT c.frames, s.difficulty_average, c.is_nomatch, s.angle
    FROM climbs c
    JOIN climb_stats s ON c.uuid = s.climb_uuid
    WHERE c.layout_id = ? 
      AND c.is_draft = 0 
      AND c.is_listed = 1
      AND s.difficulty_average IS NOT NULL
      and s.angle IS NOT NULL;
"""

cursor.execute(query, (LAYOUT_ID,))
rows = cursor.fetchall()

#get physical location of holds
#filtered for TB2 12x8 ONLY

cursor.execute("""
    SELECT p.id, h.x, h.y 
    FROM placements p 
    JOIN holes h ON p.hole_id = h.id
    WHERE h.product_id = 5 
""")
raw_coords = {str(row[0]): (row[1], row[2]) for row in cursor.fetchall()}

connection.close()

# Load or recreate the vocabulary
vocab = {"[PAD]": 0, "[START]": 1, "[END]": 2}
next_token_id = 3

tokenized_climbs = []
climb_grades = []
no_matching = []
climb_angles = []

for frame_string, grade, no_match, angle in rows:
    holds = re.findall(r'p\d+r\d+', frame_string)
    if not holds: continue

    def get_hold_coordinates(token):
        hold_id = token.split('r')[0].replace('p', '')
        if hold_id in raw_coords:
            return (raw_coords[hold_id][1], raw_coords[hold_id][0])
        return (0, 0) #fallback
    
    holds_sorted = sorted(holds, key=get_hold_coordinates, reverse=False)

    #convert each hold+colour combo to a specific number     
    for hold in holds_sorted:
        if hold not in vocab:
            vocab[hold] = next_token_id
            next_token_id += 1
            
    climb_sequence = [vocab["[START]"]] + [vocab[hold] for hold in holds_sorted] + [vocab["[END]"]]
    tokenized_climbs.append(climb_sequence)
    
    # Round the difficulty to an integer so the AI can treat it as a category (V-grade index)
    climb_grades.append(int(round(grade)))
    
    no_matching.append(no_match)

    climb_angles.append(int(angle) // 5) #increments of 5 so compress it for efficiency

#do coordinates now
min_x = min(c[0] for c in raw_coords.values())
max_x = max(c[0] for c in raw_coords.values())
min_y = min(c[1] for c in raw_coords.values())
max_y = max(c[1] for c in raw_coords.values())

vocab_coords = {}

for token_str, vocab_idx in vocab.items():
    # Give special tokens safe, neutral coordinates
    if token_str in ["[PAD]", "[START]", "[END]"]:
        vocab_coords[vocab_idx] = [0.0, 0.0]
        continue
    
    hold_id = token_str.split('r')[0].replace('p', '')
    
    if hold_id in raw_coords:
        raw_x, raw_y = raw_coords[hold_id]
        #squish coordinates safely between 0.0 and 1.0
        norm_x = (raw_x - min_x) / (max_x - min_x)
        norm_y = (raw_y - min_y) / (max_y - min_y)
        vocab_coords[vocab_idx] = [norm_x, norm_y]
    else:
        #fallback if a weird hold ID somehow slips through
        vocab_coords[vocab_idx] = [0.0, 0.0]

#write to files
suffix = ""
if LAYOUT_ID == 11:
    suffix = "_spray"
elif LAYOUT_ID == 10:
    suffix = "_mirror"
else:
    suffix = f"_layout_{LAYOUT_ID}"

with open(f'vocab{suffix}.json', 'w') as f: json.dump(vocab, f, indent=4)
with open(f'tokenized_climbs{suffix}.json', 'w') as f: json.dump(tokenized_climbs, f)
with open(f'climb_grades{suffix}.json', 'w') as f: json.dump(climb_grades, f)
with open(f'vocab_coords{suffix}.json', 'w') as f: json.dump(vocab_coords, f)
with open(f'no_matching{suffix}.json', 'w') as f: json.dump(no_matching, f)
with open(f'climb_angles{suffix}.json', 'w') as f: json.dump(climb_angles, f)

print(f"Processed {len(tokenized_climbs)} climbs")