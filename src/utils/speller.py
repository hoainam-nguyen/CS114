TOTAL_COST_chars = '0123456789-,.đdvn '
TOTAL_COST_keys = ['TỔNG TIỀN PHẢI T.TOÁN', 'Cộng tiền hàng', 'Tổng cộng', 'Tong cong', 'Tổng cộng (đã gồm VAT)',
                   'Tổng tiền', 'Tổng tiền hàng', 'TỔNG CỘNG', 'Tổng số tiền thanh toán', 'Tiền Thanh Toán',
                   'Thành Tiền', 'Total Amount', 'Total', 'Tổng thanh toán', 'Tại quầy', 'Tổng số thanh toán', 'Tổng hóa đơn (VNĐ)']

TIMESTAMP_keys = ['Ngày', 'Thời gian']
TIMESTAMP_noise_keys = ['Số HĐ', 'Số GD']

import os
from src.utils.common import cer_loss_one_image

def validate_TOTAL_COST_amount(input_str, thres=0.8):
    if len(input_str) == 0:
        return False
    count = 0
    input_str = input_str.lower()
    for ch in input_str:
        if ch in TOTAL_COST_chars:
            count += 1
    score = count / len(input_str)
    #print(score)
    return True if score > thres else False


def validate_TOTAL_COST_keys(input_str, cer_thres=0.2):
    lower_totalcost = input_str.lower()
    min_cer = 1
    for k in TOTAL_COST_keys:
        lower_k = k.lower()
        cer = cer_loss_one_image(lower_totalcost, lower_k)
        if cer < min_cer:
            min_cer = cer
    return True if min_cer < cer_thres else False


def validate_TIMESTAMP(input_str):
    lower_totalcost = input_str.lower()
    for k in TIMESTAMP_keys:
        lower_k = k.lower() +':'
        if lower_k in lower_totalcost:
            print(k, input_str)
            return True
    return False

def validate_SELLER(list_seller, input_str, cer_thres=0.2):
    if len(input_str) < 10:
        return False
    input_str = input_str.lower()
    min_cer = 1
    min_str = ''
    for s in list_seller:
        if s['count'] > 1:
            for line in s['SELLER']:
                lower_line = line.lower()
                cer = cer_loss_one_image(lower_line, input_str)
                if cer < min_cer:
                    min_cer = cer
                    min_str = lower_line
    if min_cer < cer_thres:
        print(round(min_cer, 2), input_str, '-----', min_str)
    return True if min_cer < cer_thres else False


def validate_ADDRESS(list_address, input_str, cer_thres=0.2):
    if len(input_str) < 10:
        return False
    input_str = input_str.lower()
    min_cer = 1
    min_str = ''
    for s in list_address:
        if s['count'] > 1:
            for line in s['ADDRESS']:
                lower_line = line.lower()
                cer = cer_loss_one_image(lower_line, input_str)
                if cer < min_cer:
                    min_cer = cer
                    min_str = lower_line
    if min_cer < cer_thres:
        print(round(min_cer, 2), input_str, '-----', min_str)
    return True if min_cer < cer_thres else False


def fix_datetime(input_str):  # for string len >42
    input_str = input_str.lstrip(' ').rstrip(' ')
    lower_input_str = input_str.lower()
    final_time_pos = -1
    for k in TIMESTAMP_keys:
        lower_k = k.lower()
        time_pos = lower_input_str.find(lower_k)
        if time_pos != -1:
            final_time_pos = time_pos

    final_noise_pos = -1
    for k in TIMESTAMP_noise_keys:
        lower_k = k.lower()
        noise_pos = lower_input_str.find(lower_k)
        if noise_pos != -1:
            final_noise_pos = noise_pos

    final_str = input_str
    if final_noise_pos > 0:
        final_str = input_str[:final_noise_pos]
    if final_time_pos > 0:
        final_str = input_str[final_time_pos:]

    final_str = final_str.lstrip(' ').rstrip(' ')

    return final_str

    # final_str = result_dict['TIMESTAMP'][0][time_pos:]
    # if ' -' not in final_str:
    #     final_str = final_str.replace('-', ' -')
    # if '- ' not in final_str:
    #     final_str = final_str.replace('-', '- ')
    # result_dict['TIMESTAMP'][0] = final_str



def fix_totalcost(list_totalcost, output_ocr_path = None):
    from src.utils.common import cer_loss_one_image, get_list_icdar_poly, find_TOTALCOST_val_poly
    final_list = []
    if len(list_totalcost) == 2:
        if validate_TOTAL_COST_amount(list_totalcost[0]) or validate_TOTAL_COST_keys(list_totalcost[1]):
            list_totalcost[0], list_totalcost[1] = list_totalcost[1], list_totalcost[0]
        final_list = list_totalcost
    elif len(list_totalcost) >= 3:
        # print('\n', list_totalcost)
        min_cer = 1
        min_str = ''
        for idx, totalcost in enumerate(list_totalcost):
            lower_totalcost = totalcost.lower()
            for k in TOTAL_COST_keys:
                lower_k = k.lower()
                cer = cer_loss_one_image(lower_totalcost, lower_k)
                if cer < min_cer:
                    min_cer = cer
                    min_str = totalcost

        final_list.append(min_str)

        max_len = 0
        max_str = ''
        for idx, totalcost in enumerate(list_totalcost):
            lower_totalcost = totalcost.lower()
            if validate_TOTAL_COST_amount(lower_totalcost):
                if len(totalcost) > max_len:
                    max_len = len(totalcost)
                    max_str = totalcost

        final_list.append(max_str)
        # print(img_name,final_list)
    elif len(list_totalcost)==1:
        #print(os.path.basename(output_ocr_path))
        #print(list_totalcost[0])
        # list_icdar_poly = get_list_icdar_poly(output_ocr_path)
        # list_icdar_poly = []
        # for icdar_pol in list_icdar_poly:
        #     if icdar_pol.value == list_totalcost[0] and validate_TOTAL_COST_keys(icdar_pol.value, cer_thres=0.2):
        #         TOTAL_COST_value = find_TOTALCOST_val_poly(keys_poly=icdar_pol,
        #                                 list_poly=list_icdar_poly, expand_ratio=0.2)
        #         if TOTAL_COST_value is not None:
        #             #print(TOTAL_COST_value)
        #             list_totalcost.append(TOTAL_COST_value)
        final_list = list_totalcost
    else:
        final_list = list_totalcost
    return final_list

def fix_result_by_dictionary(result_dict, dictionary):
    list_seller = dictionary['seller']
    list_address = dictionary['address']
    seller_str = ' '.join(result_dict['SELLER']['value'])
    min_cer = 1
    min_seller_string = ''
    min_seller = None
    for seller in list_seller:
        seller_ori_str = ' '.join(seller['SELLER'])
        cer = cer_loss_one_image(seller_str, seller_ori_str)
        if cer < min_cer:
            min_cer = cer
            min_seller_string = seller_ori_str
            min_seller = seller['SELLER']
            # print(min_cer, seller_ori_str)
    if min_cer < 0.3 and min_cer > 0:
        # print('Fix SELLER', seller_str, '----->', min_seller_string)
        result_dict['SELLER']['value'] = min_seller

    address_str = ' '.join(result_dict['ADDRESS']['value'])
    min_cer = 1
    min_address_string = ''
    min_address = None
    for address in list_address:
        address_ori_str = ' '.join(address['ADDRESS'])
        cer = cer_loss_one_image(address_str, address_ori_str)
        if cer < min_cer:
            min_cer = cer
            min_address_string = address_ori_str
            min_address = address['ADDRESS']
            # print(min_cer, min_address_string)
    if min_cer < 0.3 and min_cer > 0:
        # print('Fix ADDRESS', address_str, '----->', min_address_string)
        result_dict['ADDRESS']['value'] = min_address


def fix_result_by_rule_based(result_dict, ocr_path, dictionary=None):
    # Fix address than can not read
    address_to_fix = ''
    num_the_same_seller = 0
    if len(result_dict['ADDRESS']['value']) == 0 and len(result_dict['SELLER']['value']) > 0:
        full_seller = ' '.join(result_dict['SELLER']['value'])
        for store in dictionary['store']:
            if store['count'] > 10:
                full_store_seller = ' '.join(store['SELLER'])
                if cer_loss_one_image(full_seller, full_store_seller) < 0.1:
                    num_the_same_seller += 1
                    address_to_fix = store['ADDRESS']
    if num_the_same_seller == 1:
        # print(os.path.basename(ocr_path), address_to_fix)
        result_dict['ADDRESS']['value'] = address_to_fix

    # add regex datetime
    for idx, time in enumerate(result_dict['TIMESTAMP']['value']):
        if len(time) > 30:
            # print(file, result_dict['TIMESTAMP']['value'][idx], '----------------------------------->',regex)
            result_dict['TIMESTAMP']['value'][idx] = fix_datetime(time)

    # simple rule to fix order of date time
    if len(result_dict['TIMESTAMP']['value']) == 2:
        bboxes1 = result_dict['TIMESTAMP']['bboxes'][0].split(',')
        bboxes1 = [int(coor) for coor in bboxes1]
        first_center = [(bboxes1[0] + bboxes1[2] + bboxes1[4] + bboxes1[6]) / 4,
                        (bboxes1[1] + bboxes1[3] + bboxes1[5] + bboxes1[7]) / 4]
        bboxes2 = result_dict['TIMESTAMP']['bboxes'][1].split(',')
        bboxes2 = [int(coor) for coor in bboxes2]
        second_center = [(bboxes2[0] + bboxes2[2] + bboxes2[4] + bboxes2[6]) / 4,
                         (bboxes2[1] + bboxes2[3] + bboxes2[5] + bboxes2[7]) / 4]
        if second_center[0] < first_center[0]:
            # print(os.path.basename(ocr_path).replace('.txt',''))
            # print(os.path.basename(ocr_path), result_dict['TIMESTAMP']['value'])
            result_dict['TIMESTAMP']['value'][0], result_dict['TIMESTAMP']['value'][1] = \
                result_dict['TIMESTAMP']['value'][1], result_dict['TIMESTAMP']['value'][0]
            result_dict['TIMESTAMP']['bboxes'][0], result_dict['TIMESTAMP']['bboxes'][1] = \
                result_dict['TIMESTAMP']['bboxes'][1], result_dict['TIMESTAMP']['bboxes'][0]
            # print('---------------------------->',result_dict['TIMESTAMP']['value'])

    result_dict['TOTAL_COST']['value'] = fix_totalcost(result_dict['TOTAL_COST']['value'],
                                                       output_ocr_path=ocr_path)