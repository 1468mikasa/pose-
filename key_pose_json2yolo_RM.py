import os
import json
import numpy as np

bbox_class = {
    'red': 0,
    'blue': 1  # 修正后的类别映射
}

# 关键点的类别（按 group_id 顺序 0,1,2,3 对应标签 '0','1','2','3'）
keypoint_class = ['0', '1', '2', '3']

labelme_path1 = r'I:\WeChat Files\wxid_wfoauwbq0p9722\FileStorage\File\2025-04'

for file_name in os.listdir(labelme_path1):
    if file_name.endswith(".json"):
        labelme_path=os.path.join(labelme_path1,file_name)
        with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)

        labelme.keys()

        img_width = labelme['imageWidth']   # 图像宽度
        img_height = labelme['imageHeight'] # 图像高度

        # 生成 YOLO 格式的 txt 文件
        suffix = labelme_path.split('.')[-2]
        yolo_txt_path = suffix + '.txt'

        with open(yolo_txt_path, 'w', encoding='utf-8') as f:
            for each_ann in labelme['shapes']:  # 遍历每个标注

                if each_ann['shape_type'] == 'rectangle':  # 如果遇到框

                    yolo_str = ''

                    ## 框的信息
                    # 框的类别 ID
                    bbox_class_id = bbox_class[each_ann['label']]
                    yolo_str += '{} '.format(bbox_class_id)
                    # 左上角和右下角的 XY 像素坐标
                    bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                    bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                    # 框中心点的 XY 像素坐标
                    bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                    bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                    # 框宽度
                    bbox_width = bbox_bottom_right_x - bbox_top_left_x
                    # 框高度
                    bbox_height = bbox_bottom_right_y - bbox_top_left_y
                    # 框中心点归一化坐标
                    bbox_center_x_norm = bbox_center_x / img_width
                    bbox_center_y_norm = bbox_center_y / img_height
                    # 框归一化宽度
                    bbox_width_norm = bbox_width / img_width
                    # 框归一化高度
                    bbox_height_norm = bbox_height / img_height

                    yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm,
                                                                      bbox_height_norm)

                    bbox_keypoints_dict = {}
                    for ann in labelme['shapes']:  # 遍历所有标注
                        if ann['shape_type'] == 'point':
                            # 提取坐标和group_id
                            x = int(ann['points'][0][0])
                            y = int(ann['points'][0][1])
                            group_id = ann.get('group_id')
                            # 检查是否在矩形内且group_id不为None
                            if (group_id is not None and
                                    (bbox_top_left_x < x < bbox_bottom_right_x) and
                                    (bbox_top_left_y < y < bbox_bottom_right_y)):
                                bbox_keypoints_dict[group_id] = [x, y]

                    ## 按group_id顺序0-3提取关键点，缺失则补0
                    for group_id in [0, 1, 2, 3]:
                        if group_id in bbox_keypoints_dict:
                            x_norm = bbox_keypoints_dict[group_id][0] / img_width
                            y_norm = bbox_keypoints_dict[group_id][1] / img_height
                            yolo_str += f"{x_norm:.5f} {y_norm:.5f} 2 "
                        else:
                            yolo_str += "0 0 0 "  # 缺失的点补0

                    # 写入 txt 文件中
                    f.write(yolo_str + '\n')

                    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))