/************************************************************************************
***
***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "segformer.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include "ade20k.h"
#include <sys/stat.h> // for chmod()

#define MIN_HOLE_SIZE 10
#define MIN_HOLE_RATIO 0.01

static int find_closest_label(int label, std::vector<int> guide_labels);
static IMAGE* segment_to_image(TENSOR* mask);

// -----------------------------------------------------------------------------------------------
// int segment_class_color(int c)
// {
// 	int n = ARRAY_SIZE(ade20k_class_colors);
// 	return ade20k_class_colors[c % n];
// }

// char *segment_class_name(int c)
// {
// 	int n = ARRAY_SIZE(ade20k_class_names);
// 	return (char *) ade20k_class_names[c % n];
// }

static IMAGE* segment_to_image(TENSOR* mask)
{
    float* f;
    int i, j, c, n;
    IMAGE* image;

    CHECK_TENSOR(mask);
    image = image_create(mask->height, mask->width);
    CHECK_IMAGE(image);

    n = ARRAY_SIZE(ade20k_class_colors);
    for (i = 0; i < image->height; i++) {
        f = tensor_start_row(mask, 0 /*batch */, 0 /*chan */, i);
        for (j = 0; j < image->width; j++) {
            c = ade20k_class_colors[int(f[j]) % n]; // segment_class_color(int(f[j]))
            image->ie[i][j].r = ((c >> 16) & 0xff);
            image->ie[i][j].g = ((c >> 8) & 0xff);
            image->ie[i][j].b = (c & 0xff);
            image->ie[i][j].a = 255;
        }
    }

    return image;
}

static int blend_segment(TENSOR *input_tensor, TENSOR *color_tensor)
{
    check_tensor(input_tensor);
    check_tensor(color_tensor);
    check_point(input_tensor->batch == color_tensor->batch);
    check_point(input_tensor->chan == color_tensor->chan);
    check_point(input_tensor->height == color_tensor->height);
    check_point(input_tensor->width == color_tensor->width);

    int n = color_tensor->batch * color_tensor->chan * color_tensor->height * color_tensor->width;
    for (int i = 0; i < n; i++) {
        color_tensor->data[i] = 0.5*color_tensor->data[i] + 0.5*input_tensor->data[i];
    }

    return RET_OK;
}

static int find_closest_label(int label, std::vector<int> guide_labels)
{
    if (label < 0 || label >= 150)
        return 0;
    for (int i = 0; i < 150; i++) {
        int close_label = ade20k_semantic_relations[i][label];
        if (std::find(guide_labels.begin(), guide_labels.end(), close_label) != guide_labels.end())
            return close_label;
    }

    return label; // Sorry we don't find, use original
}

int segment_remove_holes(TENSOR* mask, float hole_size_ratio)
{
    int n, mask_size, max_hole_size;
    int label_count[256], label_map[256];

    check_tensor(mask);

    n = ARRAY_SIZE(label_count);
    memset(label_count, 0, n * sizeof(int));

    mask_size = mask->batch * mask->chan * mask->height * mask->width;
    max_hole_size = MAX(int(mask->height * mask->width * hole_size_ratio), MIN_HOLE_SIZE);
    syslog_debug("Segment hole size threshold = %d", max_hole_size);
    {
        for (int j = 0; j < mask_size; j++) {
            label_count[int(mask->data[j]) % n]++;
        }
    }

    // Create guide labels from big blocks
    {
        std::vector<int> guide_labels;
        for (int i = 0; i < n; i++) {
            if (label_count[i] >= max_hole_size) {
                guide_labels.push_back(i);
            }
        }
        // Remap label
        for (int i = 0; i < n; i++) {
            label_map[i] = i; // init

            if (label_count[i] > 0 && label_count[i] < max_hole_size) {
                label_map[i] = find_closest_label(i, guide_labels);
            }
        }
        guide_labels.clear();
    }

    // Update mask
    for (int j = 0; j < mask_size; j++) {
        mask->data[j] = float(label_map[int(mask->data[j]) % n]);
    }

    return RET_OK;
}

int image_segment_predict(SegmentModel *net, char *input_filename, char *output_filename)
{
    TENSOR *argv[1];

    printf("Segment %s to %s ...\n", input_filename, output_filename);
    TENSOR *input_tensor = tensor_load_image(input_filename, 0 /*alpha*/);
    check_tensor(input_tensor);

    argv[0] = input_tensor ;
    TENSOR *output_tensor = net->engine_forward(ARRAY_SIZE(argv), argv);
    check_tensor(output_tensor);

    // TENSOR *xxxx_test;
    // xxxx_test = net->get_output_tensor(">x");
    // if (tensor_valid(xxxx_test)) {
    //     tensor_show(">x", xxxx_test);
    //     tensor_destroy(xxxx_test);
    // }

    // Save result as image
    {
	    segment_remove_holes(output_tensor, MIN_HOLE_RATIO);

	    IMAGE* image = segment_to_image(output_tensor);
	    check_image(image);

        TENSOR *color_tensor = tensor_from_image(image, 0/*with_alpha*/);
        if (tensor_valid(color_tensor)) {
            blend_segment(input_tensor, color_tensor);
            tensor_saveas_image(color_tensor, 0 /*start batch*/, output_filename);
            chmod(output_filename, 0644);
            tensor_destroy(color_tensor);
        }
	    // image_save(image, output_filename);
	    image_destroy(image);
	    tensor_destroy(output_tensor);
    }

    tensor_destroy(input_tensor);

    return RET_OK;
}
