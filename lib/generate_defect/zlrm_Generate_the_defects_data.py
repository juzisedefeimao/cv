from PIL import Image
import numpy as np
from time import strftime
import os
import xml.etree.ElementTree as ET


class Generate_Defect():
    def __init__(self, save_image_root=None, save_label_root=None, read_label_root=None,
                 read_defect_root=None, save_fail_label_root=None, save_fail_image_root=None):
        self.save_image_root = save_image_root
        self.save_label_root = save_label_root
        self.read_label_root = read_label_root
        self.read_defect_root = read_defect_root
        self.fail_label_root = save_fail_label_root
        self.fail_image_root = save_fail_image_root

        self.scale_random = False
        self.ratio_random = False
        self.rotate_random = False
        self.painting_random = False
        self.translation_random = True

        # 变换后的相应缺陷图片
        self.defect_image_list = []
        self.defect_scale_image_list = []
        self.defect_ratio_image_list = []
        self.defect_rotate_image_list = []
        self.defect_translation_image_list = []
        self.defect_painting_image_list = []
        # 缺陷存放
        self.generate_defect_image_list = []

        self.defect_affirm = {'class_affirm':False, 'scale_affirm':False, 'ratio_affirm':False,
                             'rotate_affirm':False, 'painting_affirm':False}


    # 读图片，并转换为矩阵
    def readimage(self, filename, channel=None):
        image = np.array(Image.open(filename))
        if channel==1:
            image = self.image_transform_3_1(image)
        elif channel==3:
            image = self.image_transform_1_3(image)

        return image


    # 切除图片黑边
    def cutback(self, image, right_left_threshold=80, up_and_down_threshold=80):

        rows, cols = image.shape
        cols_index = cols - 1

        # 遍历判断列是否可以剪除
        def cut_rl(w_index):
            for i in range(rows):
                if image[i][w_index] > right_left_threshold:
                    return False
            return True

        # 切除右边黑边

        right_cut_x = cols_index
        while right_cut_x > 0 and cut_rl(right_cut_x):
            right_cut_x = right_cut_x - 1
        if right_cut_x == 0:
            print('图片全为黑，切除失败')
            return False
        image, _ = np.hsplit(image, (right_cut_x + 1,))

        # 切除左边黑边
        left_cut_x = 0
        print(image.shape)
        while cut_rl(left_cut_x):
            left_cut_x = left_cut_x + 1
        _, image = np.hsplit(image, (left_cut_x - 1,))

        rows_, cols_ = image.shape
        rows_index = rows_ - 1
        # 遍历判断行是否可以剪除
        def cut_ud(h_index):
            for j in range(cols_):
                if image[h_index][j] > up_and_down_threshold:
                    return False
            return True

        # 切除下边黑边
        down_cut_y = rows_index
        while cut_ud(down_cut_y):
            down_cut_y = down_cut_y - 1
        image, _ = np.split(image, (down_cut_y + 1,), axis=0)

        # 切除上边黑边
        up_cut_y = 0
        while cut_ud(up_cut_y):
            up_cut_y = up_cut_y + 1
        _, image = np.split(image, (up_cut_y - 1,), axis=0)

        print('左边切除', left_cut_x, '像素;   ', '右边切除', cols_index - right_cut_x, '像素;',
              '上边切除', up_cut_y, '像素;   ', '下边切除', rows_index - down_cut_y, '像素;')

        return image


    # 单通道图像转为3通道图像
    def image_transform_1_3(self, image):
        assert len(image.shape) != 2 or len(image.shape) != 3, print('图像既不是3通道,也不是单通道')
        if len(image.shape) == 2:
            c = []
            for i in range(3):
                c.append(image)
            image = np.asarray(c)
            image = image.transpose([1, 2, 0])
        elif len(image.shape)==3:
            print('图像为3通道图像,不需要转换')

        return image


    # 3通道图像转为单通道图像
    def image_transform_3_1(self, image):
        assert len(image.shape) != 2 or len(image.shape) != 3, print('图像既不是3通道,也不是单通道')
        if len(image.shape) == 3:
            image_2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            # 灰度化方法2：根据亮度与RGB三个分量的对应关系：Y=0.3*R+0.59*G+0.11*B
            h, w, color = image.shape
            for i in range(h):
                for j in range(w):
                    image_2[i][j] = np.uint8(0.3 * image[i][j][0] + 0.59 * image[i][j][1] + 0.11 * image[i][j][2])
            image = image_2
            assert len(image.shape) == 2, '3通道转为单通道图像失败'
        elif len(image.shape) == 2:
            print('图像为单通道图像,不需要转换')

        return image


    # 保存图片
    def saveimage(self, image, saveimage_name=None, image_ext='bmp', saveimage_root=None):
        if len(image.shape)==2:
            image = self.image_transform_1_3(image)
        if saveimage_name is None:
            saveimage_name = 'image_{}'.format(strftime("%Y_%m_%d_%H_%M_%S")) + '.' + image_ext
        else:
            saveimage_name = saveimage_name + '.' + image_ext
        if saveimage_root is None:
            saveimage_root = 'C:\\Users\\jjj\\Desktop\\jjj\\zlrm\\data\\default_root'
            print('未设置保存图片的路径，默认保存到_{}'.format(saveimage_root))
        if not os.path.isdir(saveimage_root):
            os.makedirs(saveimage_root)
        root = os.path.join(saveimage_root, str(saveimage_name))
        image = Image.fromarray(image)
        image.save(root)


    # 保存label
    def savelabel(self, boxes, labelfile, savelabel_name=None, savelabel_root=None):
        tree = ET.parse(labelfile)
        root = tree.getroot()

        if savelabel_name is None:
            savelabel_name = 'box_{}'.format(strftime("%Y_%m_%d_%H_%M_%S")) + '.' + 'x,l'
        else:
            savelabel_name = savelabel_name + '.' + 'xml'
        if savelabel_root is None:
            savelabel_root = 'C:\\Users\\jjj\\Desktop\\jjj\\zlrm\\data\\default_root'
            print('未设置保存boxes的路径，默认保存到_{}'.format(savelabel_root))
        for i in range(len(boxes)):
            # 一级
            object = ET.Element('object')
            # 二级
            name = ET.Element('name')
            name.text = boxes[i]['name']
            pose = ET.Element('pose')
            pose.text = 'Unspecified'
            truncated = ET.Element('truncated')
            truncated.text = '0'
            difficult = ET.Element('difficult')
            difficult.text = '1'
            bndbox = ET.Element('bndbox')
            # 三级
            xmin = ET.Element('xmin')
            xmin.text = str(boxes[i]['xmin'])
            ymin = ET.Element('ymin')
            ymin.text = str(boxes[i]['ymin'])
            xmax = ET.Element('xmax')
            xmax.text = str(boxes[i]['xmax'])
            ymax = ET.Element('ymax')
            ymax.text = str(boxes[i]['ymax'])
            # 将节点添加到树
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
            object.append(name)
            object.append(pose)
            object.append(truncated)
            object.append(difficult)
            object.append(bndbox)
            root.append(object)
            savelabel = os.path.join(savelabel_root, savelabel_name)
            tree.write(savelabel)

    # 生成一张纯白图片
    def generate_white_image(self, shape=(600,600)):
        image = np.zeros(shape, dtype=np.uint8)
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                image[i][j] = np.uint8(255)

        return image


    # 清空残留列表
    def clean_list(self):
        if self.defect_affirm['class_affirm']:
            self.defect_image_list = []
            self.defect_affirm['class_affirm'] = False
        if self.defect_affirm['scale_affirm']:
            self.defect_scale_image_list = []
            self.defect_affirm['scale_affirm'] = False
        if self.defect_affirm['ratio_affirm']:
            self.defect_ratio_image_list = []
            self.defect_affirm['ratio_affirm'] = False
        if self.defect_affirm['rotate_affirm']:
            self.defect_rotate_image_list = []
            self.defect_affirm['ratio_affirm'] = False
        if self.defect_affirm['painting_affirm']:
            self.defect_painting_image_list = []
            self.defect_affirm['painting_affirm'] = False


    # 为图片随机生成一些缺陷
    def generate_defects(self, image, labelfile, freehand_sketching = False, save_name=None):
        if save_name==None:
            save_name = len(os.listdir(self.save_image_root))
            save_name = save_name + 1

        if len(self.generate_defect_image_list)==0:
            for file in os.listdir(self.read_defect_root):
                if freehand_sketching and file == 'freehand_sketching':
                    freehand_sketching_folder_root = os.path.join(self.read_defect_root, 'freehand_sketching')
                    for freehand_sketching_file in os.listdir(freehand_sketching_folder_root):
                        freehand_sketching_image_root = os.path.join(freehand_sketching_folder_root,
                                                                     freehand_sketching_file)
                        freehand_sketching_image = self.readimage(freehand_sketching_image_root)
                        self.get_defect_freehand_sketching(freehand_sketching_image)
                elif file == 'paint_smear':
                    paint_smear_folder_root = os.path.join(self.read_defect_root, 'paint_smear')
                    for paint_smear_file in os.listdir(paint_smear_folder_root):
                        paint_smear_image_root = os.path.join(paint_smear_folder_root, paint_smear_file)
                        paint_smear_image = self.readimage(paint_smear_image_root)
                        self.get_defect_paint_smear(paint_smear_image)
                elif file == 'aluminium_skimmings':
                    aluminium_skimmings_folder_root = os.path.join(self.read_defect_root, 'aluminium_skimmings')
                    for aluminium_skimmings_file in os.listdir(aluminium_skimmings_folder_root):
                        aluminium_skimmings_image_root = os.path.join(aluminium_skimmings_folder_root,
                                                                      aluminium_skimmings_file)
                        aluminium_skimmings_image = self.readimage(aluminium_skimmings_image_root)
                        self.get_defect_aluminium_skimmings(aluminium_skimmings_image)
                # else:
                #     raise KeyError('未知的缺陷', file)
            # self.random_defect()
            defect_image_list = self.defect_image_list

            if self.scale_random:
                self.defect_scale(defect_image_list)
                defect_image_list = self.defect_scale_image_list
            if self.ratio_random:
                self.defect_ratio(defect_image_list)
                defect_image_list = self.defect_ratio_image_list
            if self.rotate_random:
                self.defect_rotate(defect_image_list)
                defect_image_list = self.defect_rotate_image_list
            if self.painting_random:
                self.defect_painting(defect_image_list)
                defect_image_list = self.defect_painting_image_list

            self.generate_defect_image_list = defect_image_list
            self.clean_list()

        defect_image_list = self.generate_defect_image_list
        print('生成的缺陷还有', len(defect_image_list))

        if self.translation_random:
            fetch = self.defect_translation(image, defect_image_list, labelfile)
            if fetch == None:
                print('输出未合成的label和image')
                tree = ET.parse(labelfile)
                save_xml_root = os.path.join(self.fail_label_root, save_name + '.xml')
                tree.write(save_xml_root)
                self.saveimage(image, saveimage_name=save_name, saveimage_root=self.fail_image_root)
            else:
                image = fetch[0]
                boxes = fetch[1]
                self.saveimage(image, saveimage_name=save_name, saveimage_root=self.save_image_root)
                self.savelabel(boxes, labelfile, savelabel_name=save_name, savelabel_root=self.save_label_root)


    def judge_vein_exist(self, file):
        tree = ET.parse(file)
        vein_exist = False
        for obj in tree.findall('object'):
            if obj.find('name').text == 'vein':
                vein_exist = True

        return vein_exist


    # 为一批图像生成缺陷
    def generate_defect_batch(self, batch_data_root=None):
        for labelfile in os.listdir(self.read_label_root):
            if labelfile.split('.')[-1] == 'xml':
                print('为图片 ', labelfile.split('.')[0], ' 生成缺陷')
                image_root = os.path.join(batch_data_root, labelfile.split('.')[0] + '.bmp')
                image = self.readimage(image_root, channel=1)
                # image = self.cutback(image)
                h, w = image.shape
                label_root = os.path.join(self.read_label_root, labelfile)
                if h > 200 and w > 200 and h / w < 4.4 and w / h < 4.4:
                    if self.judge_vein_exist(label_root):
                        self.generate_defects(image, label_root, save_name=labelfile.split('.')[0])
                        print('已生成', len(os.listdir(self.save_image_root)), '个图片')
                    else:
                        tree = ET.parse(label_root)
                        save_xml_root = os.path.join(self.save_label_root, labelfile.split('.')[0])
                        tree.write(save_xml_root)
                        self.saveimage(image, saveimage_name=labelfile.split('.')[0], saveimage_root=self.save_image_root)

    def preload_defect(self, preload_defect_root, freehand_sketching = False):
        for file in os.listdir(preload_defect_root):
            if freehand_sketching and file == 'freehand_sketching':
                freehand_sketching_folder_root = os.path.join(preload_defect_root, 'freehand_sketching')
                for freehand_sketching_file in os.listdir(freehand_sketching_folder_root):
                    freehand_sketching_image_root = os.path.join(freehand_sketching_folder_root,
                                                                 freehand_sketching_file)
                    freehand_sketching_image = self.readimage(freehand_sketching_image_root)
                    image = self.get_defect_freehand_sketching(freehand_sketching_image)
                    if image is not None:
                        self.saveimage(image, saveimage_name=freehand_sketching_file.split('.')[0],
                                       saveimage_root=os.path.join(self.read_defect_root, 'freehand_sketching'))

            elif file == 'paint_smear1':
                paint_smear_folder_root = os.path.join(preload_defect_root, 'paint_smear')
                for paint_smear_file in os.listdir(paint_smear_folder_root):
                    paint_smear_image_root = os.path.join(paint_smear_folder_root, paint_smear_file)
                    paint_smear_image = self.readimage(paint_smear_image_root)
                    image = self.get_defect_paint_smear(paint_smear_image, preload=True)
                    if image is not None:
                        self.saveimage(image, saveimage_name=paint_smear_file.split('.')[0],
                                       saveimage_root=os.path.join(self.read_defect_root, 'paint_smear'))

            elif file == 'aluminium_skimmings':
                aluminium_skimmings_folder_root = os.path.join(preload_defect_root, 'aluminium_skimmings')
                for aluminium_skimmings_file in os.listdir(aluminium_skimmings_folder_root):
                    aluminium_skimmings_image_root = os.path.join(aluminium_skimmings_folder_root,
                                                                  aluminium_skimmings_file)
                    aluminium_skimmings_image = self.readimage(aluminium_skimmings_image_root)
                    image = self.get_defect_aluminium_skimmings(aluminium_skimmings_image, preload=True)
                    if image is not None:
                        self.saveimage(image, saveimage_name=aluminium_skimmings_file.split('.')[0],
                                       saveimage_root=os.path.join(self.read_defect_root, 'aluminium_skimmings'))




    # 获得手绘缺陷
    def get_defect_freehand_sketching(self, image):
        if len(image.shape)==3:
            image = self.image_transform_3_1(image)
        assert len(image.shape)==2, '图片不能转为单通道'
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                if image[i][j]>200:
                    image[i][j] = 0
                else:
                    image[i][j] = 255

        image = self.cutback(image)
        if image is not False:
            print('读取缺陷完成')
            self.defect_image_list.append({'name': 'freehand_sketching', 'image': image})
            # print(len(self.defect_image))
            self.defect_affirm['class_affirm'] = True
            return image


    # 获得油污缺陷
    def get_defect_paint_smear(self, image, preload=False):
        if len(image.shape) == 3:
            image = self.image_transform_3_1(image)
        assert len(image.shape) == 2, '图片不能转为单通道'
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                if image[i][j] > 75:
                    image[i][j] = 0

        image = self.cutback(image, right_left_threshold=1, up_and_down_threshold=1)
        if image is not False:
            h, w = image.shape
            if preload:
                for i in range(h):
                    for j in range(w):
                        if image[i][j] == 0:
                            image[i][j] = 255
            print('读取缺陷完成')
            self.defect_image_list.append({'name': 'paint_smear', 'image': image})
            # print(len(self.defect_image))
            self.defect_affirm['class_affirm'] = True
            return image


    # 获得铝屑缺陷
    def get_defect_aluminium_skimmings(self, image, preload=False):
        if len(image.shape) == 3:
            image = self.image_transform_3_1(image)
        assert len(image.shape) == 2, '图片不能转为单通道'
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                if image[i][j] > 80:
                    image[i][j] = 0
        image = self.cutback(image, right_left_threshold=1, up_and_down_threshold=1)
        if image is not False:
            h, w = image.shape
            if preload:
                for i in range(h):
                    for j in range(w):
                        if image[i][j] == 0:
                            image[i][j] = 255

            print('读取缺陷完成')
            self.defect_image_list.append({'name': 'aluminium_skimmings', 'image': image})
            # print(len(self.defect_image))
            self.defect_affirm['class_affirm'] = True
            return image



    # 随机生成缺陷
    def random_defect(self, p_threshold=0.5):
        # 从一个点开始以一定的概率分布随机往外生长
        h = 0
        w = 0
        while h < 100 and w < 100:
            image = np.zeros((401, 401), dtype=np.uint8)
            h, w = image.shape
            image[0][0] = 255
            for i in range(h):
                for j in range(i + 1):
                    if j - 1 >= 0:
                        if image[j - 1][i - j] == 255:
                            if np.random.rand() < p_threshold:
                                image[j][i - j] = 255
                    if i - j - 1 >= 0:
                        if image[j][i - j - 1] == 255:
                            if np.random.rand() < p_threshold:
                                image[j][i - j] = 255
                    if j - 1 >= 0 and i - j - 1 >= 0:
                        if image[j - 1][i - j - 1] == 255:
                            if np.random.rand() < p_threshold:
                                image[j][i - j] = 255
            image = self.cutback(image)
            h, w = image.shape

        # h = 0
        # w = 0
        # while h < 100 and w < 100:
        #     image_ = np.zeros((401, 401), dtype=np.uint8)
        #     h, w = image_.shape
        #     image_[400][400] = 255
        #     for i in range(h):
        #         for j in range(i + 1):
        #             if j - 1 >= 0:
        #                 if image_[400 - j + 1][400 - i + j] == 255:
        #                     if np.random.rand() < p_threshold:
        #                         image_[400 - j][400 - i + j] = 255
        #             if i - j - 1 >= 0:
        #                 if image_[400 - j][400 - i + j + 1] == 255:
        #                     if np.random.rand() < p_threshold:
        #                         image_[400 - j][400 - i + j] = 255
        #             if j - 1 >= 0 and i - j - 1 >= 0:
        #                 if image_[400 - j + 1][400 - i + j + 1] == 255:
        #                     if np.random.rand() < p_threshold:
        #                         image_[400 - j][400 - i + j] = 255
        #     image_ = self.cutback(image_)
        #     h, w = image_.shape



        self.defect_image_list.append(image)
        # print(len(self.defect_image))
        self.defect_affirm['class_affirm'] = True
        self.saveimage(image, saveimage_name='jjj')


    # 随机的上色方案
    def painting_random_fetch(self, painting_schem=None, ):
        random = np.random.randint(1,11)
        if painting_schem == 1:
            painting = np.random.randint(1,50)
        if painting_schem == 2:
            painting = np.random.randint(70, 120)
        if painting_schem == 3:
            painting = np.random.randint(150,255)

        return painting

    # 给曲线内部上色
    def defect_painting(self, defect_image_list):
        defect_data = defect_image_list
        for n in range(len(defect_data)):
            image = defect_data[n]['image']
            h, w = image.shape
            for p in range(np.random.randint(3,5)):
                # painting_schem为随机到的上色方案，共有3套方案
                painting_schem = np.random.randint(1, 5)
                painting = 1
                if painting_schem < 4:
                    painting = self.painting_random_fetch(painting_schem=painting_schem)
                for i in range(h):
                    left_ = 0
                    left_2 = 0
                    right_ = 0
                    switch = 0
                    j = 0
                    while j < w:
                        left_2 = j
                        while j < w and image[i][j] == 0:
                            j = j + 1
                        left_ = j
                        while j < w and image[i][j] != 0:
                            j = j + 1
                        right_ = j
                        if left_ != right_:
                            if switch == 0:
                                switch = 1
                            switch = (-1)*switch
                            if switch == 1:
                                left_ = left_2

                        for k in range(left_, right_):
                            if painting_schem == 4:
                                image[i][k] = np.random.randint(1,255)
                            image[i][k] = painting
                self.defect_painting_image_list.append({'name':defect_data[n]['name'], 'image':image})
        self.defect_affirm['painting_affirm'] = True


    # 对缺陷进行旋转
    def defect_rotate(self, defect_image_list):
        defect_data = defect_image_list
        for n in range(len(defect_data)):
            image = defect_data[n]['image']
            for s in range(np.random.randint(3, 5)):
                rotation_angle = np.random.randint(0, 360)
                image = Image.fromarray(image.astype(np.uint8))
                image = image.rotate(rotation_angle)
                image = np.array(image)
                self.defect_rotate_image_list.append({'name':defect_data[n]['name'], 'image':image})
        self.defect_affirm['rotate_affirm'] = True

    # 从xml文件里得到对应铝锭表面图片的缺陷框，分为缺陷和纹理
    def get_defectbox_from_xml(self, xlm_filename):
        tree = ET.parse(xlm_filename)
        obj_box = []
        vein_box = []
        for obj in tree.findall('object'):
            if obj.find('name').text == 'vein':
                bbox = obj.find('bndbox')
                box = [int(bbox.find('xmin').text),
                       int(bbox.find('ymin').text),
                       int(bbox.find('xmax').text),
                       int(bbox.find('ymax').text)]
                vein_box.append(box)
            else:
                bbox = obj.find('bndbox')
                box = [int(bbox.find('xmin').text),
                       int(bbox.find('ymin').text),
                       int(bbox.find('xmax').text),
                       int(bbox.find('ymax').text)]
                obj_box.append(box)
        return obj_box, vein_box

    # 选择放缺陷的位置，并返回最小h， w坐标
    def select_defect_loacte(self, obj_box, vein_box, defect_size):
        # 寻找位置的次数
        find_num = 0
        vein = vein_box[np.random.randint(0, len(vein_box))]
        locate = []
        locate.append(np.random.randint(vein[1] + 1, vein[3] - defect_size[0]))#h
        locate.append(np.random.randint(vein[0] + 1, vein[2] - defect_size[1]))#w
        while self.judge_inter(obj_box, locate, defect_size) and find_num<300:
            locate[0] = np.random.randint(vein[1] + 1, vein[3] - defect_size[0])
            locate[1] = np.random.randint(vein[0] + 1, vein[2] - defect_size[1])
            find_num = find_num + 1
        if find_num < 300:
            return locate
        else:
            print('获取位置失败')
            return None

    # 判断所选的框与obj_box是否相交
    def judge_inter(self, obj_box, locate, defect_size):
        defect_box = [locate[0], locate[1], locate[0] + defect_size[1], locate[1] + defect_size[0]]
        defect_box = np.array(defect_box)
        obj_box = np.array(obj_box)
        if len(obj_box) == 0:
            inters = 0
        elif len(obj_box) == 1:
            ixmin = np.maximum(obj_box[0, 0], defect_box[0])
            iymin = np.maximum(obj_box[0, 1], defect_box[1])
            ixmax = np.minimum(obj_box[0, 2], defect_box[2])
            iymax = np.minimum(obj_box[0, 3], defect_box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
        else:
            ixmin = np.maximum(obj_box[:, 0], defect_box[0])
            iymin = np.maximum(obj_box[:, 1], defect_box[1])
            ixmax = np.minimum(obj_box[:, 2], defect_box[2])
            iymax = np.minimum(obj_box[:, 3], defect_box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
        print('inters', inters, np.sum(np.array(inters) <= 0), (np.array(inters)).size)
        if np.sum(np.array(inters) <= 0) == (np.array(inters)).size:
            return False
        else:
            return True

    # 对缺陷进行平移
    def defect_translation(self, image, defect_image_list, filename):
        # 得到缺陷的位置框和纹理的位置框
        obj_box, vein_box = self.get_defectbox_from_xml(filename)
        h, w = image.shape
        # print(len(defect_image))
        assert len(defect_image_list)>0, '未生成缺陷，不能与样本合成有缺陷的样本'
        boxes = []
        high = min(len(defect_image_list), 4)
        low = 1
        if len(defect_image_list)>=2:
            low = 2
        defect_image_fetch = np.random.randint(low=0, high=len(defect_image_list), size=np.random.randint(low, high+1))
        defect_image_fetch = list(defect_image_fetch)
        defect_image_fetch = list(set(defect_image_fetch))
        defect_image_fetch.sort(reverse=True)

        for n in defect_image_fetch:
            defect_image_ = defect_image_list[n]['image']
            defect_size = defect_image_.shape
            # print(defect_image_.shape)
            locate = self.select_defect_loacte(obj_box, vein_box, defect_size)#h,w
            if locate == None :
                return None
            else:
                for i in range(defect_size[0]):
                    for j in range(defect_size[1]):
                        if defect_image_[i][j] != 0:
                            image[i + locate[0]][j + locate[1]] = defect_image_[i][j]
                box = {'name':defect_image_list[n]['name'], 'xmin': locate[1] - 1, 'ymin': locate[0] - 1,
                       'xmax': locate[1] + defect_size[1] + 1, 'ymax': locate[0] + defect_size[0] + 1}
                print(locate)
                print('defectsize',defect_size)
                print('box',box)
                boxes.append(box)
                defect_box = [locate[1] - 1, locate[0] - 1, locate[1] + defect_size[1] + 1,
                              locate[0] + defect_size[1] + 1]
                obj_box.append(defect_box)

        for i in range(len(defect_image_fetch)):
            defect_image_list.pop(defect_image_fetch[i])

        return image, boxes


    # 按一定分布得到一随机数，以此作为缺陷图片的大小
    def scale_random_fetch(self):
        p = np.random.randint(0,10)
        if p < 2:
            size = np.random.randint(8,20)
        elif p < 4:
            size = np.random.randint(20,40)
        elif p < 6:
            size = np.random.randint(40,60)
        elif p < 8:
            size = np.random.randint(60,80)
        else:
            size = np.random.randint(80,100)
        return size


    # 对缺陷进行大小变换
    def defect_scale(self, defect_image_list):
        defect_data = defect_image_list
        for n in range(len(defect_data)):
            image = defect_data[n]['image']
            for s in range(np.random.randint(3, 5)):
                size = self.scale_random_fetch()
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((size, size), Image.ANTIALIAS)
                image = np.array(image)
                self.defect_scale_image_list.append({'name':defect_data[n]['name'], 'image':image})

        self.defect_affirm['scale_affirm'] = True


    # 对缺陷进行高宽的比例变换
    def defect_ratio(self, defect_image_list):
        defect_data = defect_image_list
        for n in range(len(defect_data)):
            image = defect_data[n]['image']
            h, w = image.shape
            for s in range(np.random.randint(3, 5)):
                h_, w_ = np.random.randint(1,11,size=2)
                size_h = np.int(np.sqrt((h * w) / (h_ * w_)) * h_) + 1
                size_w = np.int(np.sqrt((h * w) / (h_ * w_)) * w_) + 1
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((size_h, size_w), Image.ANTIALIAS)
                image = np.array(image)
                self.defect_ratio_image_list.append({'name':defect_data[n]['name'], 'image':image})
        self.defect_affirm['ratio_affirm'] = True

if __name__ == '__main__':
    # k = []
    datadir = 'H:\\defect\\paint_smear'
    ga = Generate_Defect()
    for imagefile in os.listdir(datadir):
        imageroot = os.path.join(datadir, imagefile)
        image = ga.readimage(imageroot, channel=3)
        # print(image)
        image = ga.get_defect_paint_smear(image)
        name = imagefile + 'k'
        ga.saveimage(image,saveimage_name=name, saveimage_root=datadir)
