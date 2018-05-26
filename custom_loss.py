import numpy as np
import tensorflow as tf
from keras import backend as K

def get_active_pitches_keras(pianoroll):
    """
    Given a pianoroll matrix, return a list of all pitches that 
    were played in this pianoroll.
    """
    pianoroll = K.cast_to_floatx(pianoroll)
    min_pitch = 13
    num_pitches = 96
    
    pitches = K.arange(num_pitches) + min_pitch
    active_pitch_rows = K.any(pianoroll, axis=1) # List of booleans
    active_pitches = tf.boolean_mask(pitches, active_pitch_rows)
    return active_pitches

def get_active_pitch_classes_keras(pianoroll):
    """
    Given a pianoroll matrix, return a list of all pitch classes
    (0-11 from C-B) that were played in this pianoroll.
    """
    active_pitches = get_active_pitches_keras(pianoroll)
    active_pitch_classes, _ = tf.unique(tf.floormod(active_pitches, 12))
    return active_pitch_classes

def pitch_intersection_over_union_keras(pianoroll_1, pianoroll_2):
    """
    Given two pianoroll matrices, return the intersection over union
    of their active pitch classes (ignoring octaves)
    """
    notes_1 = K.cast(get_active_pitch_classes_keras(pianoroll_1), 'int32')
    notes_2 = K.cast(get_active_pitch_classes_keras(pianoroll_2), 'int32')
    # Get intersection
    intersection = tf.sets.set_intersection(notes_1[None,:], notes_2[None,:])
    num_intersections = K.cast(tf.size(intersection.values), 'float32')
    # Get union
    union = tf.sets.set_union(notes_1[None,:], notes_2[None,:])
    num_unions = K.cast(tf.size(union.values), 'float32')
    # Get IOU
    iou = num_intersections / K.clip(num_unions, K.epsilon(), None) # Protect against 0-division
    return iou

def onsets_loss(pianorolls_batch):
    '''
    Given a batch of pianorolls (NUM_BATCHES, NUM_PITCHES, NUM_TICKS, 1),
    return a beat-loss score between -1 and 1, where -1 is perfectly on-beat.
    '''
    NUM_TICKS = 96
    # Remove channel axis
    pianorolls_batch = K.squeeze(pianorolls_batch, axis=3)
    
    # Score mask
    beats_per_unit = 4
    num_units = 1
    ticks_per_beat = 24
    sigma = 2
    score_mask_row = -np.ones(NUM_TICKS) # Num ticks
    for half_beat in range(2*beats_per_unit*num_units):
        hb = half_beat * ticks_per_beat / 2
        next_hb = hb + ticks_per_beat / 2
        # Good ticks
        score_mask_row[hb : hb + sigma + 1] = 1
        score_mask_row[next_hb - sigma : next_hb] = 1
        # Impartial ticks
        score_mask_row[hb + ticks_per_beat / 4] = 0
    score_mask = np.zeros(K.eval(pianorolls_batch).shape)
    score_mask[:,:] = score_mask_row # Fill all rows

    note_onset_matrix = get_note_onsets_keras(pianorolls_batch)
    score = K.sum(note_onset_matrix * K.constant(score_mask))
    # Normalize score by number of notes
    num_notes = K.sum(note_onset_matrix)
    norm_score = score / K.clip(num_notes, K.epsilon(), None) # Protect against 0-division
    # Invert so that this is a loss that we minimize
    return -norm_score

def get_note_onsets_keras(pianorolls_batch):
    '''
    Given a batch of pianorolls (NUM_BATCHES, NUM_PITCHES, NUM_TICKS),
    return a similar-shaped batch of 1 or 0 indicating onset or not.
    '''
    pianorolls_batch = K.cast(pianorolls_batch, 'float32')
    # Binarize
    binarized = K.sign(pianorolls_batch)
    # Pad along time axis (axis=2)
    padding = K.zeros((pianorolls_batch.shape[0], pianorolls_batch.shape[1],1))
    padded = K.concatenate([padding, binarized, padding], axis=2)
    # Diff
    diff = padded[:,:,1:] - padded[:,:,:-1]
    # Keep only positive values
    note_ons = K.greater(diff, 0)
    # Discard last column
    note_ons = note_ons[:,:,:-1]
    return K.cast(note_ons, 'float32')

def smoothness_loss(pianorolls_batch):
    EPSILON = 1e-4
    # Remove channel axis
    pianorolls_batch = K.squeeze(pianorolls_batch, axis=3)
    # Take difference along time axis
    diff = pianorolls_batch[:,:,1:] - pianorolls_batch[:,:,:-1]
    # Smoothness as ratio of nonzero changes over all changes
    epsilon_diff = K.greater(diff, EPSILON)
    num_nonzero = K.cast(tf.count_nonzero(epsilon_diff, axis=(1,2)), 'float32')
    num_elements = K.cast(tf.size(diff[0]), 'float32')
    smoothness = num_nonzero / num_elements
    # Average over all batches
    mean_smoothness = K.mean(smoothness)
    return mean_smoothness

### METRICS
# These are essentially just wrappers around the loss functions,
# used for on-training measurement

def onsets_metric(y_true, y_pred):
    return onsets_loss(y_pred)

def smoothness_metric(y_true, y_pred):
    return smoothness_loss(y_pred)