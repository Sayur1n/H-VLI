import os
import json
import random
import base64

def load_and_sample_data(file_path, sample_size=None, seed=42, manual_ratios=None):
    """
    Load data and perform sampling.
    :param manual_ratios: dict, e.g., {'101': 0.1, '002': 0.2, ...}. If not None, ratio-based sampling is enabled.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def get_distribution(dataset):
        dist = {}
        for _, item in dataset.items():
            t = str(item.get("type", "unknown"))
            dist[t] = dist.get(t, 0) + 1
        return dist

    if not sample_size or sample_size <= 0 or sample_size >= len(data):
        print(f"📋 Loading all {len(data)} samples (No sampling).")
        final_counts = get_distribution(data)
        return data, final_counts 

    print(f"🎲 Sampling {sample_size} items with seed {seed}...")
    random.seed(seed)

    grouped = {}
    for sid, item in data.items():
        t = str(item.get("type", "unknown"))
        if t not in grouped:
            grouped[t] = []
        grouped[t].append((sid, item))
    
    unique_types = list(grouped.keys())
    
    selected_items = []
    remaining_pool = []

    if manual_ratios:
        print(f"⚙️  Using Manual Ratios: {manual_ratios}")
        total_ratio = sum(manual_ratios.values())
        
        for t in unique_types:
            items_in_group = grouped[t]
            random.shuffle(items_in_group)
            
            ratio = manual_ratios.get(t, 0.0)
            target_count = int(sample_size * (ratio / total_ratio)) if total_ratio > 0 else 0
            
            take_count = min(len(items_in_group), target_count)
            selected_items.extend(items_in_group[:take_count])
            remaining_pool.extend(items_in_group[take_count:])

    else:
        print("⚖️  Using Auto-Balanced Sampling")
        if len(unique_types) > 0:
            quota_per_type = sample_size // len(unique_types)
        else:
            quota_per_type = sample_size
            
        for t in unique_types:
            items_in_group = grouped[t]
            random.shuffle(items_in_group)
            take_count = min(len(items_in_group), quota_per_type)
            selected_items.extend(items_in_group[:take_count])
            remaining_pool.extend(items_in_group[take_count:])

    needed = sample_size - len(selected_items)
    if needed > 0:
        print(f"   Filling gap of {needed} samples from remaining pool...")
        if len(remaining_pool) >= needed:
            selected_items.extend(random.sample(remaining_pool, needed))
        else:
            selected_items.extend(remaining_pool)

    random.shuffle(selected_items)
    
    final_data_map = dict(selected_items)
    final_counts = get_distribution(final_data_map)
    
    print(f"✅ Final sampled count: {len(selected_items)}")

    return final_data_map, final_counts

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_full_path(base_dir, relative_path):
    return os.path.join(base_dir, relative_path)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_txt(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
