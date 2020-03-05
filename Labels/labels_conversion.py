import numpy as np
import pickle

def from_onehot_to_binary(label):
    '''
    label: two-entries binary array
    '''
    try:
        return label[0]
    except:
        print('The label must be a two-entries array')

def from_binary_to_onehot(label):
    '''
    label: binary digit
    '''
    if label == 0:
        onehot = np.array([0,1])
    elif label == 1:
        onehot = np.array([1,0])
    return onehot

def zeros_ones_counter(binary_vector):
    len_vector = len(binary_vector)
    zeros_count = 0
    for x in binary_vector:
        if x == 0:
            zeros_count += 1
    return zeros_count, len_vector-zeros_count

def from_single_frame_to_multi_frames_label(data, multi_frame_size = 5, threshold = 0.5):
    '''
    data: array of size (n,2), where n is the number of frames
    multi_frame_size: size of the input clip
    threshold: value between 0 and 1
    return an array of size (n/multi_frame_size,2)
    '''
    count = 0
    tmp_label_vect = np.array([])
    multi_frames_labels = np.array([0,0])
    for label in data:
        if count < multi_frame_size:
            try:
                tmp_label_vect = np.append(tmp_label_vect, from_onehot_to_binary(label))
                count += 1
            except:
                print('Error')
        if count == multi_frame_size:
            zeros, ones = zeros_ones_counter(tmp_label_vect)
            if zeros == 0 or ones/zeros > threshold:
                multi_frames_labels = np.vstack([multi_frames_labels,from_binary_to_onehot(1)])
            else:
                multi_frames_labels = np.vstack([multi_frames_labels,from_binary_to_onehot(0)])
            count = 0
            tmp_label_vect = np.array([])
    return multi_frames_labels[1:]
            
if __name__ == "__main__":
    with open('../Binary Outputs/chievo_juve_1_binary.pkl', 'rb') as f:
        data = pickle.load(f)
    multi_frame_label = from_single_frame_to_multi_frames_label(data)
    print(len(multi_frame_label))