import os
import json

bbox_class = {
    'red': 0,
    'blue': 1
}

# 关键点标签与索引映射（按顺序0-3）
keypoint_class = ['0', '1', '2', '3']

labelme_path1 = r'I:\WeChat Files\wxid_wfoauwbq0p9722\FileStorage\File\2025-04'  # 替换为实际路径

for file_name in os.listdir(labelme_path1):
    if file_name.endswith(".json"):
        labelme_path = os.path.join(labelme_path1, file_name)
        with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)

        img_width = labelme['imageWidth']
        img_height = labelme['imageHeight']

        # 生成 YOLO 格式的 txt 文件
        suffix = labelme_path.split('.')[-2]
        yolo_txt_path = suffix + '.txt'

        with open(yolo_txt_path, 'w', encoding='utf-8') as f:
            for each_ann in labelme['shapes']:
                if each_ann['shape_type'] == 'rectangle':
                    yolo_str = ''

                    # 矩形框类别
                    bbox_class_id = bbox_class[each_ann['label']]
                    yolo_str += f"{bbox_class_id} "

                    # 矩形框坐标（保留浮点）
                    x1 = min(each_ann['points'][0][0], each_ann['points'][1][0])
                    y1 = min(each_ann['points'][0][1], each_ann['points'][1][1])
                    x2 = max(each_ann['points'][0][0], each_ann['points'][1][0])
                    y2 = max(each_ann['points'][0][1], each_ann['points'][1][1])

                    # 计算中心点归一化坐标
                    cx = (x1 + x2) / 2 / img_width
                    cy = (y1 + y2) / 2 / img_height
                    w = (x2 - x1) / img_width
                    h = (y2 - y1) / img_height
                    yolo_str += f"{cx:.5f} {cy:.5f} {w:.5f} {h:.5f} "

                    # 收集属于该框的关键点
                    bbox_keypoints = {}
                    for point_ann in labelme['shapes']:
                        if point_ann['shape_type'] == 'point':
                            point_label = point_ann['label']
                            if point_label not in keypoint_class:
                                continue  # 跳过无效标签

                            # 将标签映射为group_id（0-3）
                            group_id = keypoint_class.index(point_label)

                            # 关键点坐标（保留浮点）
                            px = point_ann['points'][0][0]
                            py = point_ann['points'][0][1]

                            # 判断是否在框内
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                # 归一化坐标
                                px_norm = px / img_width
                                py_norm = py / img_height
                                bbox_keypoints[group_id] = [px_norm, py_norm]

                    # 按0-3顺序填充关键点
                    for group_id in range(4):
                        if group_id in bbox_keypoints:
                            px, py = bbox_keypoints[group_id]
                            yolo_str += f"{px:.5f} {py:.5f} 2 "  # 2表示可见
                        else:
                            yolo_str += "0 0 0 "  # 缺失补零

                    f.write(yolo_str.strip() + '\n')
                    print(f'{labelme_path} --> {yolo_txt_path} 转换完成')