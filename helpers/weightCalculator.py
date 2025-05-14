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

<<<<<<< Updated upstream
def compute_score(row):
    weights = POSITION_FEATURE_WEIGHTS.get(row['position'], {})
    return sum(row.get(feat, 0) * weight for feat, weight in weights.items())
=======
# General stats used in all roles
GENERAL_STATS = {
    'avg_minutes': 1,
    'man_of_the_match': SIGNIFICANCE_WEIGHTS['high'],
    'yellow_cards': SIGNIFICANCE_WEIGHTS['penalty'],
    'red_cards': SIGNIFICANCE_WEIGHTS['penalty'] * 2,
    'own_goals': SIGNIFICANCE_WEIGHTS['penalty'] * 2
}

# OFF stats
OFF_STATS = {
    'goals': SIGNIFICANCE_WEIGHTS['high'],
    'assists': SIGNIFICANCE_WEIGHTS['high'],
    'shots_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'fouled_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'offsides_per_game': SIGNIFICANCE_WEIGHTS['penalty'],

    'fouls': SIGNIFICANCE_WEIGHTS['moderate'],
    'crosses': SIGNIFICANCE_WEIGHTS['moderate'],
    'pass_success_pct': SIGNIFICANCE_WEIGHTS['moderate'],
    'dribbles_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'dispossessed_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'bad_control_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'key_passes_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'through_balls_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'aerials_won_per_game': SIGNIFICANCE_WEIGHTS['moderate'],

    'interceptions_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'offside_won_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'clearances_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'dribbled_past_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'outfielder_block_per_game': SIGNIFICANCE_WEIGHTS['insignificant']
}

# MID stats
MID_STATS = {
    'assists': SIGNIFICANCE_WEIGHTS['high'],
    'key_passes_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'passes_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'pass_success_pct': SIGNIFICANCE_WEIGHTS['high'],
    'through_balls_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'dribbles_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'aerials_won_per_game': SIGNIFICANCE_WEIGHTS['high'],

    'shots_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'dispossessed_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'bad_control_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'fouled_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'fouls': SIGNIFICANCE_WEIGHTS['moderate'],
    'interceptions_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'tackles': SIGNIFICANCE_WEIGHTS['moderate'],

    'clearances_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'offsides_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'offside_won_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'dribbled_past_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'outfielder_block_per_game': SIGNIFICANCE_WEIGHTS['insignificant']
}

# DEF stats
DEF_STATS = {
    'tackles': SIGNIFICANCE_WEIGHTS['high'],
    'interceptions_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'clearances_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'aerials_won_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'outfielder_block_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'offside_won_per_game': SIGNIFICANCE_WEIGHTS['high'],
    'dribbled_past_per_game': SIGNIFICANCE_WEIGHTS['high'],

    'pass_success_pct': SIGNIFICANCE_WEIGHTS['moderate'],
    'bad_control_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'dispossessed_per_game': SIGNIFICANCE_WEIGHTS['moderate'],
    'fouls': SIGNIFICANCE_WEIGHTS['moderate'],

    'goals': SIGNIFICANCE_WEIGHTS['insignificant'],
    'assists': SIGNIFICANCE_WEIGHTS['insignificant'],
    'shots_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'key_passes_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'crosses': SIGNIFICANCE_WEIGHTS['insignificant'],
    'through_balls_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'dribbles_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'offsides_per_game': SIGNIFICANCE_WEIGHTS['insignificant'],
    'fouled_per_game': SIGNIFICANCE_WEIGHTS['insignificant']
}

# Compute general component
def compute_general_score(stats):
    avg_minutes = stats['minutes_played'] / stats['apps'] if stats['apps'] > 0 else 0
    return (
        avg_minutes * GENERAL_STATS['avg_minutes'] +
        stats['man_of_the_match'] * GENERAL_STATS['man_of_the_match'] +
        stats['yellow_cards'] * GENERAL_STATS['yellow_cards'] +
        stats['red_cards'] * GENERAL_STATS['red_cards'] +
        stats['own_goals'] * GENERAL_STATS['own_goals']
    )

# Generalized score function
def compute_role_score(stats, role_weights):
    score = compute_general_score(stats)
    for key, weight in role_weights.items():
        if key in stats:
            score += stats[key] * weight
    return score

# Role-specific functions
def compute_off_score(stats):
    return compute_role_score(stats, OFF_STATS)

def compute_mid_score(stats):
    return compute_role_score(stats, MID_STATS)

def compute_def_score(stats):
    return compute_role_score(stats, DEF_STATS)
>>>>>>> Stashed changes
