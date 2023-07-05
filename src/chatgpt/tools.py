import os
import openai

from dotenv import load_dotenv, find_dotenv

# Load .env
_ = load_dotenv(find_dotenv())

openai.api_key=os.environ["OPENAI_API_KEY"]

system_message_linelevel = '''
Your task is to extract invoice information from user data.
The user input message will be delimited with ### characters.
Provide your output in json format with the keys: SELLER, ADDRESS, TIMES, COST

- SELLER: Store name, proper noun | return -1 if none
- ADDRES: Store address | return -1 if none
- TIMES: Invoice time | return -1 if none
- COST: The total charge payable includes information such as, "Tổng hóa đơn", "Tổng cộng", "THANH TOÁN", "TỔNG TIỀN PHẢI T. TOÁN", | return -1 if none

EXAMPLE 1: user input ###
VM4 QNH 112 Thanh Niên
Số 112 Thanh Niên, P. Cầm Thành
UnCommert
TP. Câm Phả, Quảng Ninh
024.71068868-39671
HÓA ĐƠN BÁN HẰNG
Ngày bán: 11/08/2020 12:46 | HD:00181055
Quầy:001 | NVBH:09015348
Mặt hàng | Đơn giá | T.TTiển | SL
Đậu hũ miếng Bibigo ông Kim's 250gr
89385085471 | 13.200 | 2 | 26.400
TỔNG TIẾN PHẢI T. TOÁN | 26.400
TỔNG TIỀN ĐÃ GIẢM | O
TIỀN KHÁCH | 1 | 26.400
TIÊN MẶT | 26.400
TIỀN TRÃ LẠI
đã bao gồm thuế gtgt) | (Gi
Chi xuất hóa đơn trong ngày
in 19 19 then same tain same day | Taxiivviiced
CÁM DN QUÝ KHÁCH VÀ HỆN GẶP LẠI
1066866 Website: Vinmart.com | Hotline: 02
###

Your response: ###
{
    "SELLER": "UnCommert",
    "ADDRESS": "Số 112 Thanh Niên, P. Cầm Thành",
    "TIMES": "Ngày bán: 11/08/2020 12:46",
    "COST": "TỔNG TIẾN PHẢI T. TOÁN | 26.400"
}

EXAMPLE 2: user input ###
Saigon Co.op
Co.op Food HN THE K-PARK
Ma so thue: 0309129418?115
Tang 1,5H42 - Toa K3,Cong trinh chung cu
ket hop dich vu tren o dat H-CT2 thuoc
Du an Dau tu xay dung Khu nha o Hi Brand
tai Khu do thi moi Van Phu,
P. Phu La, 0. Ha Dong, Tp Ha Noi
Email: cfthekparkecoopfood.vn
Website: www.saigonco-op.com.vn
PHIẾU TINH TIEN
Co gia tri xuat Hoa don GTGT trong ngay
Quay: 02 | Ngay: 23/09/2019 11:12:15
Nhan vien: Nguyen Viet Hieu So HD: 09348
00000002953005 Than ngoai bo kg
VATO57 | 336,000.00 | 39,648.00
0.118 KG
00000002950181 Bong cai xanh Kg
VATO57 | 50,500.00 | 14,241.00
10.282 KG
Tong so tien thanh toan: | 53,889.000
Tong so luong hang: | 2
Phuong thuc thanh toan:
Tien mat | 60,000.00
Tien thoi lai cac cho | -6,111.00
Bao gom thue GTGT 57: | 2,566.14
THONG TIN KHÁCH HANG THAN THIET:
Ma 50 1 | 2800272077
Ho ten: (Bong) 9124. DAO THI HUYEN
Diem dau ngay: | 143
Diem su dung trong hoa don:
Tran trong cam on Quy Khach da dung nguyen nam
hang tai Co.opFood. Hen gap lai
Ma HD: EOEUACNVOANMY 11:13 27938 | 27938
###

Your response: ###
{
    "SELLER": "Saigon Co.op",
    "ADDRESS": "P. Phu La, 0. Ha Dong, Tp Ha Noi",
    "TIMES": "11:13",
    "COST": "Tong so tien thanh toan: | 53,889.000"
}

'''


system_message_wordlevel = '''
Your task is to extract invoice information from user data.
The user input message will be delimited with ### characters.
Provide your output in json format with the keys: SELLER, ADDRESS, TIMES, COST_NAME, COST_VALUE

- SELLER: Store name, proper noun | return -1 if none
- ADDRES: Store address ( must be meaningful alf an address) | return -1 if none
- TIMES: Invoice time | return -1 if none
- COST_NAME:  Filed shows the total payment name, such as "Tổng hóa đơn", "Tổng cộng", "Thanh toán", "TỔNG TIỀN PHẢI T. TOÁN" | return -1 if none
- COST_VALUE: Filed shows the total payment value, such as 300.000 VND, 15.VND (usually the largest value)| return -1 if none

EXAMPLE 1: user input ###

###
LCCOFFEE
65 Làng Tăng Phú, Phường Tăng Nhơn Phú An
Quận 9, TP Hồ Chí Minh
0902002187-0944938811
HÓA ĐƠN THANH TOÁN
Số: 100011248
Thanh toán luôn
Thu ngân: NHAN VIEN
Giờ vào: 11:46 28/10/2022
Giờ in: 11:46
Mặt hàng
SL/TL THÀNH TIỀN
Bạc Siu/White Coffe (Iced)
35,000
Tiền hàng (1)
35,000
THANH TOÁN
35,000 d
Tiền mặt
35,000 đ
Cảm ơn quý khách và hẹn gặp lại "
Wifi: LC COFFEE
Mật khẩu: 68686868
###

your response: ###
{
    "SELLER": "LCCOFFEE",
    "ADDRESS": "65 Làng Tăng Phú, Phường Tăng Nhơn Phú An",
    "TIMES": "Giờ vào: 11:46 28/10/2022",
    "COST_NAME": "THANH TOÁN",
    "COST_VALUE": "35,000"
}

Example 2: user input ###
VM4 QNH 1060-1062 Trân Phú Tr
Số 1060-1062 Đường Trần Phú
Vincommerce
TP. Câm Phà, Quảng Ninh
02471086856-49301
HÓA ĐN BÁN HÀNG
H:00064608
Ngày bán: 15/08/2020 11:04
NVBH:09024144
Quầy:001
T.TTiển
dn giá
Mặt hàng
SL
Trứng gà tươi Dabaco hộp 10 quả
25.600
08936057560032
25.600
TỔNG TIỀN PHẢI T. TOÁN
25.600
TỔNG TIẾN Ả GIẢM
0
TIỀN KHÁCH TRẢN
25.600
TIỀN MẶT
25.600
TIỀN TRẢ LẠI T
O
(Giá đã bao gồm thuế gtgt)
Chỉ xuất hóa đơn trong ngày
Tax invoice will be issued within same day
CÁM DN QUÝ KHÁCH VÀ HỆN GẶP LẠI
Hotline: 02471066866 Website: vinmart.com
###

your respose: ###
{
    "SELLER": "Vincommerce",
    "ADDRESS": "Số 1060-1062 Đường Trần Phú, TP. Câm Phà, Quảng Ninh",
    "TIMES": "Ngày bán: 15/08/2020 11:04",
    "COST_NAME": "TỔNG TIỀN PHẢI T. TOÁN",
    "COST_VALUE": "25.600"
}
'''

class ChatTooLs():
    system_message_linelevel = system_message_linelevel
    system_message_wordlevel = system_message_wordlevel
    def get_delimiters():
        delimiters = "###"
        return delimiters

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_completion(system_message, 
        user_prompt, 
        model="gpt-3.5-turbo", 
        temperature=0, 
        max_tokens=500
    ):
        
        # define message
        delimiters = '###'
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"{delimiters}{user_prompt}{delimiters}"}]
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens,
        )

        content = response.choices[0].message["content"]
        token_dict = {
            'prompt_tokens':response['usage']['prompt_tokens'],
            'completion_tokens':response['usage']['completion_tokens'],
            'total_tokens':response['usage']['total_tokens'],
        }

        return content, token_dict
    

if __name__=="__main__":
    import time
    import json 

    sample = "tmp/text.txt"
    content = open(sample, 'r').read()
    response = ChatTooLs.get_completion(
        system_message=system_message_wordlevel, user_prompt=content
    )
    print(response[0])


    exit()
    root_path = "/mlcv/WorkingSpace/Personals/namnh/2023/Projects/CS338/namnh/Receipt-Information-Extraction-main/result/"
    files = os.listdir(root_path)

    print("Starting")
    st = time.time()

    list_file_outs = os.listdir("results")
    i = 0
    for fn in files:
        fout = fn.replace('txt', 'json')
        if fout in list_file_outs:
            continue

        sample = os.path.join(root_path, fn)
        content = open(sample, 'r').read()

        try:
            response = ChatTooLs.get_completion(
                system_message=system_message_linelevel, user_prompt=content
            )
        except Exception as err:
            print(err)
            total_time = time.time()
            mean_time = (total_time - st)/i
            print("Mean time used:", mean_time)
            exit()

        i += 1
        output = dict(content=response[0], token_dict=response[1])
        print(response[0])
        print(response[1])

        json.dump(output, open(f'results/{fout}', 'w'), indent=4, ensure_ascii=False)


    total_time = time.time()
    mean_time = (total_time - st)/i
    print("Mean time used:", mean_time)