import numpy as np
from numpy.linalg import norm

def compute_windowed_cosine_similarity(seq1, seq2, window_size=15):
    """
    Computes a windowed cosine similarity score between two sequences,
    scaled by the ratio of their vector magnitudes to penalize differences in intensity.
    """
    if not seq1 or not seq2:
        return 0.0

    s1 = np.array(seq1, dtype=np.float32)
    s2 = np.array(seq2, dtype=np.float32)

    # Pad the shorter sequence to match the length of the longer one
    len_diff = len(s1) - len(s2)
    if len_diff > 0:
        s2 = np.pad(s2, (0, len_diff), 'constant', constant_values=np.mean(s2) if s2.size > 0 else 0)
    elif len_diff < 0:
        s1 = np.pad(s1, (0, -len_diff), 'constant', constant_values=np.mean(s1) if s1.size > 0 else 0)

    if len(s1) < window_size:
        return 0.0

    scores = []
    for i in range(len(s1) - window_size + 1):
        window1 = s1[i:i + window_size]
        window2 = s2[i:i + window_size]

        norm1 = norm(window1)
        norm2 = norm(window2)
        
        # 1. Calculate the cosine similarity for the SHAPE
        if norm1 == 0 or norm2 == 0:
            cos_sim = 0.0
        else:
            cos_sim = np.dot(window1, window2) / (norm1 * norm2)
        
        # Scale shape similarity from [-1, 1] to [0, 1]
        shape_score = (cos_sim + 1) / 2
        
        # 2. Calculate the scaling factor for MAGNITUDE
        # This penalizes if one performance is "bigger" than the other
        magnitude_scaler = 0.0
        if norm1 > 1e-6 and norm2 > 1e-6:
             magnitude_scaler = min(norm1, norm2) / max(norm1, norm2)

        # 3. Combine the scores
        # The final score is high only if both shape and magnitude are similar
        combined_score = shape_score * magnitude_scaler
        scores.append(combined_score)

    if not scores:
        return 0.0

    # Average the combined scores and scale to 0-100
    final_score = np.mean(scores) * 100
    return round(final_score, 1)

def compute_mae_similarity_score(seq1_aligned: list, seq2_aligned: list, max_angle_range: float = 180.0):
    if not seq1_aligned or not seq2_aligned or len(seq1_aligned) == 0: return 0.0
    s1, s2 = np.array(seq1_aligned), np.array(seq2_aligned)
    mae = np.mean(np.abs(s1 - s2))
    normalized_error = min(mae / max_angle_range, 1.0)
    return round((1.0 - normalized_error) * 100.0, 1)

def compute_correlation_similarity_score(seq1_aligned: list, seq2_aligned: list):
    if not seq1_aligned or not seq2_aligned or len(seq1_aligned) < 2: return 0.0
    s1, s2 = np.array(seq1_aligned), np.array(seq2_aligned)
    std1, std2 = np.std(s1), np.std(s2)
    if std1 < 1e-6 and std2 < 1e-6: return 100.0 if np.abs(np.mean(s1) - np.mean(s2)) < 1e-5 else 0.0
    if std1 < 1e-6 or std2 < 1e-6: return 0.0
    try: corr = np.corrcoef(s1, s2)[0, 1]
    except Exception: return 0.0
    if np.isnan(corr): return 0.0
    return round(((corr + 1.0) / 2.0) * 100.0, 1)