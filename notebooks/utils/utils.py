import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from PIL import Image as pil_image
import jsonlines
import numpy as np
import requests
import json
import os

def get_image_segmentations(aks_service, img_path):

    # URL for the web service
    scoring_uri = aks_service.scoring_uri

    # If the service is authenticated, set the key or token
    key, _ = aks_service.get_keys()

    # Load image data
    data = open(img_path, "rb").read()

    # Set the content type
    headers = {"Content-Type": "application/octet-stream"}

    # If authentication is enabled, set the authorization header
    headers["Authorization"] = f"Bearer {key}"

    # Make the request and display the response
    resp = requests.post(scoring_uri, data, headers=headers)
    # print(resp.text)
    return resp

def generate_jsonl_annotations(source, target_path, annotation_file):
    annotations = []
    # delete annotation flie if it exists
    with open(annotation_file, 'w') as fp:
        fp.close()

    # loop through images
    for img_idx, image in enumerate(source['images']):

        id = image['id']
        width = image['width']
        height = image['height']
        file_name = image['file_name']
        extension = file_name.split('.')[-1].lower()

        # get class from image file name 
        class_name = file_name.split('_')[0]
        
        image_dict = {
            "image_url" : target_path + '/' + class_name + '/' + file_name,
            "image_details" : {
                "format" : extension,
                "width" : width,
                "height" : height },
            "label" : []
        }
        
        # get all annotations for current image
        image_annotations = [annotation for annotation in source['annotations'] if annotation['image_id'] == id]
        
        label = {}

        # loop through annotations
        for anno_idx, annotation in enumerate(image_annotations):

            iscrowd = annotation['iscrowd']
            # processing normal cases (iscrowd is 0):
            if iscrowd == 0:
            
                polygons = []
                # loop through list of polygons - will be 1 in most cases
                for segmentation in annotation['segmentation']:
                    
                    polygon = []
                    # loop through vertices:
                    for id, vertex in enumerate(segmentation):
                        if (id % 2) == 0:
                            # x-coordinates (even index)
                            x = vertex / width
                            polygon.append(x)
                
                        else:
                            y = vertex / height
                            polygon.append(y)
                    polygons.append(polygon)
            
                image_dict['label'].append({
                    "label" : class_name,
                    "isCrowd" : iscrowd,
                    "polygon" : polygons
                })
            # TODO: process iscrowd annotations
            if iscrowd != 0:
                pass
        
        # write entry to JSONL file. 
        with jsonlines.open(annotation_file, mode='a') as writer:
                writer.write(image_dict)


def plot_ground_truth_boxes(image_file, ground_truth_boxes):
    # Display the image
    plt.figure()
    img_np = mpimg.imread(image_file)
    img = pil_image.fromarray(img_np.astype("uint8"), "L")
    img_w, img_h = img.size
    x, y = img.size

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.imshow(img_np, cmap='gray')
    ax1.axis("off")

    ax2.imshow(img_np, cmap='gray')
    ax2.axis("off")

    fig.suptitle(ground_truth_boxes[0]["label"])

    for gt in ground_truth_boxes:
        label = gt["label"]
        # print(gt)

        polygon = gt['polygon']

        # plt.text(topleft_x, topleft_y - 10, label, color=color, fontsize=20)

        # xmin, ymin, xmax, ymax =  gt["topX"], gt["topY"], gt["bottomX"], gt["bottomY"]
        # topleft_x, topleft_y = img_w * xmin, img_h * ymin
        # width, height = img_w * (xmax - xmin), img_h * (ymax - ymin)

        color = 'mediumseagreen'

        polygon_np = np.array(polygon[0])
        polygon_np = polygon_np.reshape(-1, 2)
        polygon_np[:, 0] *= x
        polygon_np[:, 1] *= y
        poly = patches.Polygon(polygon_np, True, facecolor=color, alpha=0.4)
        ax2.add_patch(poly)
        poly_line = Line2D(polygon_np[:, 0], polygon_np[:, 1], linewidth=1, color='white')
                        #    marker='o', markersize=1, markerfacecolor='white')

        ax2.add_line(poly_line)

    plt.show()

def plot_ground_truth_boxes_jsonl(image_file, jsonl_file):
    image_base_name = os.path.basename(image_file)
    ground_truth_data_found = False
    with open(jsonl_file) as fp:
        for line in fp.readlines():
            line_json = json.loads(line)
            filename = line_json["image_url"]
            if image_base_name in filename:
                ground_truth_data_found = True
                plot_ground_truth_boxes(image_file, line_json["label"])
                break
    if not ground_truth_data_found:
        print("Unable to find ground truth information for image: {}".format(image_file))

def plot_ground_truth_boxes_dataset(image_file, dataset_pd):
    image_base_name = os.path.basename(image_file)
    image_pd = dataset_pd[dataset_pd['portable_path'].str.contains(image_base_name)]
    if not image_pd.empty:
        ground_truth_boxes = image_pd.iloc[0]["label"]
        plot_ground_truth_boxes(image_file, ground_truth_boxes)
    else:
        print("Unable to find ground truth information for image: {}".format(image_file))

def plot_predicted_segmentations(sample_image, jsonl_file, resp):

    # IMAGE_SIZE = (18,20)
    plt.figure()

    img_np=mpimg.imread(sample_image)
    img = pil_image.fromarray(img_np.astype('uint8'),'L')
    x, y = img.size

    # fig,ax = plt.subplots(1, figsize=(15,15))
    # # Display the image
    # ax.imshow(img_np, cmap='gray')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    ax1.imshow(img_np, cmap='gray')
    ax1.title.set_text('Original image')
    ax1.axis("off")

    ax2.imshow(img_np, cmap='gray')
    ax2.title.set_text('Ground truth segmentations')
    ax2.axis("off")

    ax3.imshow(img_np, cmap='gray')
    ax3.title.set_text('Predicted segmentations')
    ax3.axis("off")


    # draw box and label for each detection 
    detections = json.loads(resp.text)
    labels = [detect['label'] for detect in detections['boxes']]
    # print(labels)

    image_base_name = os.path.basename(sample_image)
    ground_truth_data_found = False
    with open(jsonl_file) as fp:
        for line in fp.readlines():
            line_json = json.loads(line)
            # print(line_json)
            filename = line_json["image_url"]
            if image_base_name in filename:
                ground_truth_data_found = True
                ground_truth_boxes = line_json["label"]
                break
    if not ground_truth_data_found:
        print("Unable to find ground truth information for image: {}".format(image_file))

    fig.suptitle(ground_truth_boxes[0]["label"])

    for gt in ground_truth_boxes:
        label = gt["label"]
        polygon = gt['polygon']

        # plt.text(topleft_x, topleft_y - 10, label, color=color, fontsize=20)

        # xmin, ymin, xmax, ymax =  gt["topX"], gt["topY"], gt["bottomX"], gt["bottomY"]
        # topleft_x, topleft_y = img_w * xmin, img_h * ymin
        # width, height = img_w * (xmax - xmin), img_h * (ymax - ymin)

        color = 'mediumseagreen'

        polygon_np = np.array(polygon[0])
        polygon_np = polygon_np.reshape(-1, 2)
        polygon_np[:, 0] *= x
        polygon_np[:, 1] *= y
        poly = patches.Polygon(polygon_np, True, facecolor=color, alpha=0.4)
        ax2.add_patch(poly)
        poly_line = Line2D(polygon_np[:, 0], polygon_np[:, 1], linewidth=1, color='white')
                        #    marker='o', markersize=1, markerfacecolor='white')

        ax2.add_line(poly_line)

    for detect in detections['boxes']:
        label = detect['label']
        box = detect['box']
        polygon = detect['polygon']
        conf_score = detect['score']
        if conf_score > 0.6:
            ymin, xmin, ymax, xmax =  box['topY'],box['topX'], box['bottomY'],box['bottomX']
            topleft_x, topleft_y = x * xmin, y * ymin
            width, height = x * (xmax - xmin), y * (ymax - ymin)
            # print('{}: [{}, {}, {}, {}], {}'.format(detect['label'], round(topleft_x, 3), 
            #                                         round(topleft_y, 3), round(width, 3), 
            #                                         round(height, 3), round(conf_score, 3)))

            color = 'mediumseagreen'
            # rect = patches.Rectangle((topleft_x, topleft_y), width, height, 
            #                         linewidth=2, edgecolor=color,facecolor='none')

            # ax.add_patch(rect)
            # plt.text(topleft_x, topleft_y - 10, label, color='black', fontsize=10)
            
            polygon_np = np.array(polygon[0])
            polygon_np = polygon_np.reshape(-1, 2)
            polygon_np[:, 0] *= x
            polygon_np[:, 1] *= y
            poly = patches.Polygon(polygon_np, True, facecolor=color, alpha=0.4)
            ax3.add_patch(poly)
            poly_line = Line2D(polygon_np[:, 0], polygon_np[:, 1], linewidth=1, color='white')
                            # marker='o', markersize=2, markerfacecolor=color)
            ax3.add_line(poly_line)

    plt.show()
