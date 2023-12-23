import torch
import numpy as np
import pdb

class SegReMapping:
    def __init__(self, mapping_name, min_ratio=0.01):
        self.label_mapping = np.load(mapping_name)
        self.min_ratio = min_ratio
        self.label_ipt = []
        # (Pdb) self.label_mapping -- (150, 150)
        # array([[ 32,  25,  97, ...,  86,  86,  80],
        #        [ 97,  86,  82, ...,  97,  97,  97],
        #        [ 86,  97, 136, ...,  21,  80,  43],
        #        ...,
        #        [111, 118,  62, ..., 149,  54,  54],
        #        [118, 135, 118, ...,  54, 111, 118],
        #        [  0,   1,   2, ..., 147, 148, 149]])

    def cross_remapping(self, content_seg, style_seg):
        cont_label_info = []
        new_cont_label_info = []
        for label in np.unique(content_seg):
            cont_label_info.append(label)
            new_cont_label_info.append(label)

        style_label_info = []
        new_style_label_info = []
        for label in np.unique(style_seg):
            style_label_info.append(label)
            new_style_label_info.append(label)

        cont_set_diff = set(cont_label_info) - set(style_label_info)
        cont_set_diff = set(cont_set_diff) - set(self.label_ipt)
        # Find the labels that are not covered by the style
        # Assign them to the best matched region in the style region
        for s in cont_set_diff:
            cont_label_index = cont_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in style_label_info:
                    new_cont_label_info[cont_label_index] = new_label
                    break
        new_content_seg = content_seg.copy()
        for i, current_label in enumerate(cont_label_info):
            new_content_seg[(content_seg == current_label)] = new_cont_label_info[i]
        return new_content_seg


    def self_remapping(self, seg):
        init_ratio = self.min_ratio
        # Assign label with small portions to label with large portion
        new_seg = seg.copy()
        [h,w] = new_seg.shape
        n_pixels = h*w
        # First scan through what are the available labels and their sizes
        label_info = []
        ratio_info = []
        new_label_info = []
        for label in np.unique(seg):
            ratio = np.sum(np.float32((seg == label))[:])/n_pixels
            label_info.append(label)
            new_label_info.append(label)
            ratio_info.append(ratio)
        for i, current_label in enumerate(label_info):
            if ratio_info[i] < init_ratio:
                for j in range(self.label_mapping.shape[0]):
                    new_label = self.label_mapping[j,current_label]
                    if new_label in label_info:
                        index = label_info.index(new_label)
                        if index >= 0:
                            if ratio_info[index] >= init_ratio:
                                new_label_info[i] = new_label
                                break
        for i, current_label in enumerate(label_info):
            new_seg[(seg == current_label)] = new_label_info[i]
        return new_seg

class TorchSegReMapping:
    def __init__(self, mapping_name, min_ratio=0.01):
        self.min_ratio = min_ratio
        self.label_mapping = torch.from_numpy(np.load(mapping_name))
        # (Pdb) self.label_mapping -- (150, 150)
        # tensor([[ 32,  25,  97, ...,  86,  86,  80],
        #        [ 97,  86,  82, ...,  97,  97,  97],
        #        [ 86,  97, 136, ...,  21,  80,  43],
        #        ...,
        #        [111, 118,  62, ..., 149,  54,  54],
        #        [118, 135, 118, ...,  54, 111, 118],
        #        [  0,   1,   2, ..., 147, 148, 149]])
        # torch.int64

    def cross_remapping(self, content_seg, style_seg):
        # regard style has big holes, but content has small holes
        unique_content_labels = torch.unique(content_seg)
        unique_style_labels = torch.unique(style_seg)

        new_content_seg = content_seg.clone()
        new_unique_content_labels = unique_content_labels.clone()

        for hole in new_unique_content_labels:
            new_hole = self.find_closest_hole(hole, unique_style_labels)
            new_unique_content_labels[unique_content_labels == hole] = new_hole

        for i, label in enumerate(new_unique_content_labels):
            new_content_seg[content_seg == label] = new_unique_content_labels[i]

        return new_content_seg


    def find_closest_hole(self, small_hole_label, big_holes):
        candidate_sets = self.label_mapping[:, small_hole_label]
        for label in candidate_sets:
            if label in big_holes: # OK we find 
                return label
        return small_hole_label # Sorry we dont find


    def self_remapping(self, seg):
        # replace small hole with closest big hole
        h, w = seg.shape
        min_n_pixels = max(int(h * w * self.min_ratio), 10)

        new_seg = seg.clone()
        unique_labels, unique_counts = torch.unique(seg, return_counts=True)
        small_holes = unique_labels[unique_counts < min_n_pixels]
        big_holes = unique_labels[unique_counts >= min_n_pixels]

        new_unique_labels = unique_labels.clone()
        for hole in small_holes:
            new_hole = self.find_closest_hole(hole, big_holes)
            new_unique_labels[unique_labels == hole] = new_hole

        for i, label in enumerate(unique_labels):
            new_seg[seg == label] = new_unique_labels[i]

        return new_seg
