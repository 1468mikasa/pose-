import os
import math
import cv2
import random
import numpy as np
# 新增的依赖
import imgaug.augmenters as iaa
import imageio



def MotionBlur(image, in_m, kernel_size=15, angle=0, gailv=0.5):
    number = random.randint(0, 10)
    if number <= gailv * 10:
        # 生成运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        center = (kernel_size - 1) // 2
        angle_rad = math.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 在核中绘制运动轨迹
        for i in range(kernel_size):
            delta = i - center
            x = int(round(center + delta * cos_angle))
            y = int(round(center + delta * sin_angle))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1

        # 避免全零核
        if kernel.sum() == 0:
            return image, in_m

        kernel /= kernel.sum()  # 归一化
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image, in_m
    else:
        return image, in_m
def open_txt(txt_file_path):
    with open(txt_file_path, 'r') as file:
        # 读取文件内容
        file_content = file.read()

        data_text = f'''{file_content}'''
        # 处理文本数据并存储为矩阵
        matrix = []
        for line in data_text.split('\n'):
            row = [float(x) if x else 0.0 for x in line.split(' ')]
            matrix.append(row)
    return matrix

#点绕中心旋转
def rotate_point_around_center(x, y, cx, cy, angle):
    # 将角度转换为弧度
    angle_rad = math.radians(-angle)
    # 将坐标平移，使得旋转中心成为原点
    x -= cx
    y -= cy
    # 计算旋转后的坐标（相对于旋转中心）
    x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    # 将坐标平移回原来的坐标系
    x_new += cx
    y_new += cy
    return (x_new, y_new)

#旋转增强方法
def xuanzhuan(image,in_m,alofa,gailv):
    number = random.randint(0, 10)
    if(number<=gailv*10):
        # 获取图像的尺寸
        (h, w) = image.shape[:2]

        # 定义旋转中心，通常是图像的中心点
        center = (w // 2, h // 2)

        # 获取旋转矩阵，旋转角度为a，缩放因子为1
        M = cv2.getRotationMatrix2D(center, alofa, 1)

        # 执行仿射变换（旋转）
        rotated = cv2.warpAffine(image, M, (w, h))
        angle=alofa
        out_m=[[0, 0] for _ in range(in_m.__len__())]
        for i in range(in_m.__len__()):
            if (in_m[i][0] != 0 or in_m[i][1] != 0):
                out_m[i][0],out_m[i][1]=rotate_point_around_center(in_m[i][0]*w, in_m[i][1]*h, w/2, h/2, angle)
                out_m[i][0]=out_m[i][0]/w
                out_m[i][1]=out_m[i][1]/h

        return rotated,out_m
    else:
        return image,in_m
#翻转
def ShangXiaFan(image,in_m,gailv):

    number = random.randint(0, 10)
    if(number<=gailv*10):
        flip_vertical = cv2.flip(image, 0)
        for i in range(in_m.__len__()):
            if (in_m[i][0] != 0 or in_m[i][1] != 0):
                in_m[i][1]=1-in_m[i][1]
        return flip_vertical,in_m
    else:
        return image,in_m
def ZuoYouFan(image,in_m,gailv):

    number = random.randint(0, 10)
    if(number<=gailv*10):
        flip_vertical = cv2.flip(image, 1)
        for i in range(in_m.__len__()):
            if(in_m[i][0]!=0 or in_m[i][1]!=0):
                in_m[i][0]=1-in_m[i][0]
        #cv2.imwrite("Fan",)
        return flip_vertical,in_m
    else:
        return image,in_m

#放大图片
def ReSize(image,in_md,beishu,gailv):
    number = random.randint(0, 10)
    if(number<=gailv*10):
        # 获取图片的尺寸
        height, width = image.shape[:2]
        # 图片的中心点
        center_x = width / 2
        center_y = height / 2
        # 获取旋转矩阵，旋转角度为a，缩放因子为1
        M = cv2.getRotationMatrix2D((center_x,center_y), 0, beishu)

        # 执行仿射变换（旋转）
        resized_image = cv2.warpAffine(image, M, (width, height))


        md_out = [[0, 0] for _ in range(in_md.__len__())]
        for i in range(in_md.__len__()):
            if (in_md[i][0] != 0 or in_md[i][1] != 0):
                x=in_md[i][0]*width
                y=in_md[i][1]*height
                # 点到中心点的距离
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # 放大后的距离
                expanded_distance = beishu * distance

                # 放大后的坐标

                expanded_x = center_x + (expanded_distance * ((x - center_x) / distance))
                expanded_y = center_y + (expanded_distance * ((y - center_y) / distance))
                md_out[i][0]=expanded_x/width
                md_out[i][1] = expanded_y/height

        return resized_image,md_out
    else:
        return image,in_md

#亮度和对比度
def ReDL(image,gailv):
    number = random.randint(0, 10)

    if(number<=gailv*10):
        alpha = random.randint(90, 150)/100  # 对比度控制 (1.0-3.0)
        beta = random.randint(-5, 25)# 亮度控制 (0-100)
        # 调整亮度和对比度
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return adjusted
    else:return image



in_txt_file = r'I:\WeChat Files\wxid_wfoauwbq0p9722\FileStorage\File\2025-04'
in_img_file = r'I:\WeChat Files\wxid_wfoauwbq0p9722\FileStorage\File\2025-04'

out_txt_file = r'I:\WeChat Files\wxid_wfoauwbq0p9722\FileStorage\File\2025-04'
out_img_file = r'I:\WeChat Files\wxid_wfoauwbq0p9722\FileStorage\File\2025-04'

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(out_img_file):
    os.makedirs(out_img_file)

if not os.path.exists(out_txt_file):
    os.makedirs(out_txt_file)

for file_name in os.listdir(in_txt_file):
    if not file_name.endswith("classes.txt"):
        for chongfu in range(1):
            try:
                txt_file_path = os.path.join(in_txt_file, file_name)
                m1 = open_txt(txt_file_path)
                m_len = len(m1)
                rule_len = 0
                for k in range(m_len):
                    if len(m1[k]) >= 17:
                        rule_len += 1

                # 获取 n*四点
                md = [[0.0, 0.0] for _ in range(4 * rule_len)]

                j = 0
                for k in range(rule_len):
                    for i in range(4):
                        j = i * 3
                        if j <= 4 * 3 - 3:
                            md[i + 4 * k][0] = m1[k][j + 5]
                            md[i + 4 * k][1] = m1[k][j + 6]
                    j = 0

                # 完整图片路径
                img_name = file_name.replace('.txt', '.jpg')
                img_path = os.path.join(in_img_file, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    # 完整图片路径
                    img_name = file_name.replace('.txt', '.png')
                    img_path = os.path.join(in_img_file, img_name)
                    img = cv2.imread(img_path)
                if img is None:
                    print(f"错误：无法加载图像 {img_path}")
                    continue

                #把框的2点加入md再取出
                for k in range(m_len):
                    if len(m1[k]) < 5:
                        print(f"无效标注行: 文件 {file_name} 第 {k} 行数据不足")
                        continue  # 跳过该行
                    w=m1[k][3]
                    h=m1[k][4]
                    kuangx1=m1[k][1]-w/2
                    kuangy1=m1[k][2]-h/2
                    kuangx2=m1[k][1]+w/2
                    kuangy2=m1[k][2]+h/2
                # 新数据
                    new_data = [[kuangx1, kuangy1], [kuangx2, kuangy2]]
                # 将新数据添加到 mdx 中
                    md.extend(new_data)
                # 增强图片和数据
                image, md = ZuoYouFan(img, md, 0.5)
                image, md = xuanzhuan(image, md, random.randint(-5, 5), 0.5)
                image, md = ShangXiaFan(image, md, -0.5)
                image = ReDL(image, 0.5)
                # 新增运动模糊（添加到增强链末尾）
            #     image, md = MotionBlur(
            #        image,
            #        md,
            #        kernel_size=random.choice([5, 10]),  # 随机核大小
            #        angle=random.randint(-5, 5),  # 随机方向
            #        gailv=0.95  # 50%概率应用
            #    )
                _angle_ = random.randint(0, 360)
                _k_ = random.randint(3, 15)
                motion_blur_aug = iaa.MotionBlur(k=_k_, angle=_angle_)  # 调整k和angle参数

                image = motion_blur_aug.augment_image(image)  # 注意这里返回单张图像

                image, md = ReSize(image, md, 0.5 + random.randint(0, 50) / 100, 0.5)

                # 取出后两组数据
                mdx=[]
                for d in range(rule_len):
                    new_data = [md.pop(), md.pop()]
                    mdx.append(new_data)

                # 完整的文件路径
                img_path = os.path.join(out_img_file, f"{chongfu}-" + img_name)
                cv2.imwrite(img_path, image)
                txt_path = os.path.join(out_txt_file, f"{chongfu}-" + file_name)

                # 写入文本内容到文件
                with open(txt_path, 'a') as file:
                    a = len(md) / 4
                    for k in range(int(a)):
                        x_max = y_max = 0
                        x_min = y_min = 100000

                        if mdx[k][0][0]< mdx[k][1][0]:
                                x_max = mdx[k][1][0]
                                x_min = mdx[k][0][0]
                        else:
                                x_max = mdx[k][0][0]
                                x_min = mdx[k][1][0]
                        if mdx[k][0][1] < mdx[k][1][1]:
                                y_max=mdx[k][1][1]
                                y_min=mdx[k][0][1]
                        else:
                                y_max=mdx[k][0][1]
                                y_min=mdx[k][1][1]


                        file.write(f"{int(m1[k][0])} {x_min + (x_max - x_min) / 2:.16f} {y_min + (y_max - y_min) / 2:.16f} {x_max - x_min:.16f} {y_max - y_min:.16f}")
                        for i in range(4):
                            file.write(f" {md[i + k * 4][0]:.16f} {md[i + k * 4][1]:.16f} 2")
                        file.write(f"\n")
                print("转换", file_name)
                del img
                del image

            except Exception as e:
                print(f"处理文件 {file_name} 时发生错误: {str(e)}")
                continue  # 跳过当前循环，继续处理下一个文件

    else:
        print("无法转换", file_name)