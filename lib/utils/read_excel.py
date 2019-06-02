import xlrd
import os
import numpy as np
save_root = 'D:\\jjj\\zlrm\\data\\mini_imagenet\\datasets\\main'

def read_excel(file):
    sheet = {}
    wb = xlrd.open_workbook(filename=file)#打开文件
    for sheet_name in wb.sheet_names():
        p_sheet = wb.sheet_by_name(sheet_name)
        if p_sheet.nrows>0 and p_sheet.ncols>0:
            sheet[sheet_name] = p_sheet

    for sheet_index in sheet:
        one_sheet = sheet[sheet_index]

        for i in range(one_sheet.nrows):
            name_txt = os.path.join(save_root, one_sheet.cell_value(i, 1) + '_' + sheet_index + '.txt')
            with open(name_txt, 'a') as f:
                f.write(str(one_sheet.cell_value(i, 0)).split('.')[0] + '\n')

if __name__ == '__main__':
    file = 'D:\\jjj\\zlrm\\data\\mini_imagenet\\datasets\\main\\test.xlsx'
    read_excel(file)