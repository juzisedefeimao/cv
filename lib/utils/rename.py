import os
import shutil
import xml.etree.ElementTree as ET

data_dir = 'D:\\jjj\\zlrm\\data\\siammask_data\\datasets\\ImageSetsClassify'
xml_dir = 'D:\\jjj\\zlrm\\data\\siammask_data\\AnnotationsClassify'
save_xml_dir = 'D:\\jjj\\zlrm\\data\\siammask_data\\Annotations'
save_root = 'D:\\jjj\\zlrm\\data\\siammask_data\\datasets\\main'
image_dir = 'D:\\jjj\\zlrm\\data\\siammask_data\\datasets\\ImageSets'
# data_dir = 'D:\\jjj\\zlrm\\data\\classifier_data\\datasets\\jjj'

def rename():
    for classes_folder in os.listdir(data_dir):


        classes_data_dir = os.path.join(data_dir, classes_folder)
        i = 1
        for classes_data in os.listdir(classes_data_dir):
            classes_data_name = os.path.join(classes_data_dir, classes_data)
            classes_data_new_name = os.path.join(classes_data_dir, classes_folder + '_' + str(i) + '_new.bmp')
            os.rename(classes_data_name, classes_data_new_name)
            i = i + 1

def save_main():
    for classes_folder in os.listdir(data_dir):
        if classes_folder.split('_')[-1] != 'test' and classes_folder.split('_')[-1] != 'val':
            classes_data_dir = os.path.join(data_dir, classes_folder)
            for classes_data in os.listdir(classes_data_dir):
                name_txt = os.path.join(save_root, classes_folder + '_train' + '.txt')
                with open(name_txt, 'a') as f:
                    f.write(classes_data.split('.')[0] + '\n')
        else:
            classes_data_dir = os.path.join(data_dir, classes_folder)
            for classes_data in os.listdir(classes_data_dir):
                name_txt = os.path.join(save_root, classes_folder + '.txt')
                with open(name_txt, 'a') as f:
                    f.write(classes_data.split('.')[0] + '\n')

def save_main_():
    label = ['vein', 'slag_inclusion', 'aluminium_skimmings', 'crack', 'edge_crack', 'paint_smear', 'dirty', 'dirty_bag']
    label_defect = ['slag_inclusion', 'aluminium_skimmings', 'crack', 'edge_crack', 'paint_smear', 'dirty', 'dirty_bag']

    file_list_t = []
    write_file_train = os.path.join(save_root, 'train.txt')
    write_file_val = os.path.join(save_root, 'test.txt')
    write_file_t = open(write_file_train, "w")
    write_file_v = open(write_file_val, "w")
    i = 1
    for file in os.listdir(save_xml_dir):
        if file.endswith(".xml"):
            lablefilename = os.path.join(save_xml_dir, file)
            tree = ET.parse(lablefilename)
            objects = False
            for obj in tree.findall('object'):
                assert obj.find('name').text in label, '有异常标签'
                if obj.find('name').text in label_defect:
                    objects = True
            if objects:
                write_name = file.split('.')[0]
                file_list_t.append(write_name)
                i = i + 1
    print(i - 1)

    sorted(file_list_t)
    number_of_lines_t = len(file_list_t)

    for current_line in range(number_of_lines_t):
        if current_line < number_of_lines_t:
            write_file_t.write(file_list_t[current_line] + '\n')
        else:
            write_file_v.write(file_list_t[current_line] + '\n')

    write_file_t.close()
    write_file_v.close()
    print('success')

def save_image():
    for classes_folder in os.listdir(data_dir):
        classes_data_dir = os.path.join(data_dir, classes_folder)
        for classes_data in os.listdir(classes_data_dir):
            classes_data_name = os.path.join(classes_data_dir, classes_data)
            classes_data_new_name = os.path.join(image_dir, classes_folder + '_' + classes_data)
            shutil.copyfile(classes_data_name, classes_data_new_name)

def save_xml():
    for classes_folder in os.listdir(xml_dir):
        classes_data_dir = os.path.join(xml_dir, classes_folder)
        for classes_data in os.listdir(classes_data_dir):
            classes_data_name = os.path.join(classes_data_dir, classes_data)
            classes_data_new_name = os.path.join(save_xml_dir, classes_folder + '_' + classes_data)
            shutil.copyfile(classes_data_name, classes_data_new_name)

def file_num():
    num = {}
    for classes_folder in os.listdir(data_dir):
        classes_data_dir = os.path.join(data_dir, classes_folder)
        num[classes_folder] = len(os.listdir(classes_data_dir))

    print(num)
    num = sorted(num[i] for i in num)
    print(num)

if __name__ == '__main__':
    # save_image()
    # save_xml()
    # save_main()
    save_main_()