import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import losses
from tensorflow import keras as tfK

def get_pitch_class_histogram(pianorolls_batch):
    """
    Given a batch of pianoroll matrices, return a Tensor of shape (NUM_BATCHES, 12)
    with each 12 elements in a batch being the sum of velocities for that pitch class, 
    normalized between zero and one.
    """
    pianorolls_batch_tensor = K.cast(K.squeeze(pianorolls_batch, -1), 'float32')
    # Get sum of velocities for each pitch class
    pitch_velocities = K.sum(pianorolls_batch_tensor, axis=2) # sum along tick axis
    pitch_class_velocities = K.sum(K.reshape(pitch_velocities, (-1, 8, 12)), axis=1) # Separated into octaves
    # Normalize against total note velocities in this pianoroll
    total_velocities = K.sum(pitch_class_velocities, axis=1, keepdims=True)
    norm_pitch_class_velocities = pitch_class_velocities / total_velocities
    return norm_pitch_class_velocities

def pitch_histogram_distance(pianorolls_batch_1, pianorolls_batch_2, distance_metric='cosine'):
    """
    Calculates the pitch class histograms for two batches of pianoroll matrices,
    then using cosine proximity to calculate the distance between histograms.
    Returns a distance value between 0 and 1, where 0 is maximum similarity and 
    1 is completely orthogonal.
    """
    hist1 = get_pitch_class_histogram(pianorolls_batch_1)
    hist2 = get_pitch_class_histogram(pianorolls_batch_2)
    if distance_metric == 'cosine':
        distance = 1 + losses.cosine_proximity(hist1, hist2)
    elif distance_metric == 'mse':
        distance = losses.mean_squared_error(hist1, hist2)
    return distance

def get_active_pitch_classes_keras(pianorolls_batch):
    """
    Given a batch of pianoroll matrices, return a boolean Tensor of shape
    (NUM_BATCHES, 12) indicating whether a particular pitch class was played 
    in this pianoroll.
    """
    EPSILON = 1e-2
    pianorolls_batch = K.cast(pianorolls_batch, 'float32')

    epsilon_active = K.greater(pianorolls_batch, EPSILON)
    active_pitch_rows = K.any(epsilon_active, axis=2) # List of booleans
    active_pitch_rows = K.reshape(active_pitch_rows, (-1, 8, 12)) # Separated into octaves
    # Get active pitches
    active_pitch_classes = K.any(active_pitch_rows, axis=1)
    return active_pitch_classes

def pitch_intersection_over_union_keras(pianorolls_batch_1, pianorolls_batch_2):
    """
    Given two batches of pianoroll matrices, return the intersection over union
    of their active pitch classes (ignoring octaves)
    """
    notes_1 = get_active_pitch_classes_keras(pianorolls_batch_1)
    notes_2 = get_active_pitch_classes_keras(pianorolls_batch_2)
    # Join both matrices
    notes_1 = tfK.backend.expand_dims(notes_1)
    notes_2 = tfK.backend.expand_dims(notes_2)
    notes_concat = K.concatenate([notes_1, notes_2])
    # Get intersection
    intersections = K.cast(K.all(notes_concat, axis=-1), 'float32')
    num_intersections = K.sum(intersections, axis=-1)
    # Get union
    unions = K.cast(K.any(notes_concat, axis=-1), 'float32')
    num_unions = K.sum(unions, axis=-1)
    # Calculate average IOU across all batches
    iou = K.mean(num_intersections / K.clip(num_unions, K.epsilon(), None)) # Protect against 0-division
    return iou

def pitch_loss(pianorolls_batch_1, pianorolls_batch_2):
    """
    Pitch loss is 1 when IOU is 0, and 0 when IOU is 1. 
    """
    return 1 - pitch_intersection_over_union_keras(pianorolls_batch_1, pianorolls_batch_2)

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
    score_mask = np.zeros((1024, 96, NUM_TICKS))#K.int_shape(pianorolls_batch))
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
    padding = K.zeros((1024, 96, 1)) #(pianorolls_batch.shape[0], pianorolls_batch.shape[1],1))
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
    clipped_diff = K.clip(diff, 0, EPSILON)
    return K.mean(clipped_diff)

#     # Smoothness as ratio of nonzero changes over all changes
#     epsilon_diff = K.greater(diff, EPSILON)
    
#     num_nonzero = K.cast(tf.count_nonzero(epsilon_diff, axis=(1,2)), 'float32')
#     num_elements = K.cast(tf.size(diff[0]), 'float32')
#     smoothness = tf.divide(num_nonzero, num_elements)
#     # Average over all batches
#     mean_smoothness = K.mean(smoothness)
#     return mean_smoothness

### METRICS
# These are essentially just wrappers around the loss functions,
# used for on-training measurement

def onsets_metric(y_true, y_pred):
    return onsets_loss(y_pred)

def smoothness_metric(y_true, y_pred):
    return smoothness_loss(y_pred)