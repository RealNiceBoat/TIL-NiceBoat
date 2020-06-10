# TIL-NiceBoat

## Problem Statement
Your job is to create 2 models to facilitate the robot's detection. The first part is in processing the written description of the victim, otherwise known as Natural Language Processing (NLP). Once provided the statement, the robot should be able to break it down to classify the words and identify various aspects key to saving our victim, such as their gender as well as their clothes.

The second problem is the detection of our victim using Computer Vision (CV). For this task our robot should be able to take in a picture, that may or may not contain our victim, and be able to draw bounding boxes indicating where in the picture there are clothes, as well as what kind of clothes they are, such as trousers or dresses. From there on, if the clothes in the box match up to the written description, the robot will know that it has identified the victim, and will proceed to save it using its arm.

## Natural Language Processing (NLP) Dataset Breakdown
To prevent manual labelling on the dataset, the original text has been masked. In the resources, we have provided a training dataset, test dataset, a modified masked GloVe word embeddings.

### Text Example:
`“He was walking out on the streets when someone spilled coffee onto his blue cashmere jacket and khaki trousers”`

### Masked words:
`“w11 w53 w243 w465 w6754 w3463 w6 w65344 w322 w9823 w33453 w84 w4356 w2234 w1246 w98302 w551 w8785 w344”`

### Labels
Training and test csv format

`id, outwear, top, trousers, women dresses, women skirts`

`0, 1, 0, 1, 0, 0`

Interpretation of above format:
For id 0, the labels are:
outwear: 1
top: 0
trousers: 1
women dresses: 0
women skirts: 0

## Computer Vision (CV) Fashion Dataset Breakdown
There are a total of 8,225 training images and 1,474 validation images.

### Labels
Each training and test JSON is similar to the COCO data format:

`{ "images": [...], "annotations": [...], "categories": [...] }`

Example of JSON:

`"images":[{ "file_name": "1.jpg" "id": "1" }]`

`"annotations":[{ "id": 1, "image_id": 10, "category_id": 4, "bbox": [704, 620, 1401, 1645] # left, top, width, height (xywh) }]`

`"categories": [ {"id": 1, "name": "tops"}, {"id": 2, "name": "trousers"}, {"id": 3, "name": "outerwear"}, {"id": 4, "name": "dresses"}, {"id": 5, "name": "skirts"}]}`
