POSITION_FEATURE_WEIGHTS = {
    'OFF': {
        'finishing': 3,
        'creativity': 3,
        'distribution': 1,
        'defense': 1,
        'duels': 2
    },
    'MID': {
        'finishing': 2,
        'creativity': 3,
        'distribution': 3,
        'defense': 2,
        'duels': 2
    },
    'DEF': {
        'finishing': 1,
        'creativity': 1,
        'distribution': 2,
        'defense': 4,
        'duels': 3
    }
}

#def compute_score(row):
    #weights = POSITION_FEATURE_WEIGHTS.get(row['position'], {})
    #return sum(row.get(feat, 0) * weight for feat, weight in weights.items())

def compute_score(row):
    if row.get('off', 0) == 1:
        weights = POSITION_FEATURE_WEIGHTS['OFF']
    elif row.get('mid', 0) == 1:
        weights = POSITION_FEATURE_WEIGHTS['MID']
    elif row.get('def', 0) == 1:
        weights = POSITION_FEATURE_WEIGHTS['DEF']
    else:
        weights = {}

    score = sum(row.get(feat, 0) * weight for feat, weight in weights.items())
    return round(score, 2)

