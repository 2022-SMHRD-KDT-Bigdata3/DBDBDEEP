import math
import os
import cx_Oracle
from datetime import datetime

def calDist():
    # 디렉토리 경로 지정
    dir_path = "./runs/detect/exp/labels/"
    LOCATION = r"C:\Users\smhrd\yolov5\instantclient_21_9"
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]
    dngr = {'2' : 'middle_glasses',
        '3' : 'middle_knife',
        '4' : 'low_pot',
        '5' : 'high_microwave',
        '6' : 'high_oven',
        '7' : 'middle_vase',
        '8' : 'middle_scissors',
        '9' : 'high_highlight',
        '10' : 'high_stove',
        '11' : 'high_powerSocket',
        '12' : 'high_gasStove',
        '13' : 'low_curtain',
        '14' : 'low_windowBlind'}
    # 디렉토리 내 모든 파일 리스트
    file_list = os.listdir(dir_path)
    if not file_list:
        print("No files in the directory")
        return
    else:
        # print(file_list)
        last_file = file_list[-1]
        # print(last_file)
        

        with open(os.path.join(dir_path, last_file), 'r') as f:
            last_file_content = f.read()


        last_file_content_splitList = last_file_content.split()

        class_list = []

        for i in range(len(last_file_content_splitList)):
            if(i%5==0):
                class_list.append(last_file_content_splitList[i])

        dngr_list = ['2','3','4','5','6','7','8','9','10','11','12','13','14']
        dngr_class = []
        pet_list = []
        if('0' in class_list or '1' in class_list):
            for i in range(len(dngr_list)):
                if(dngr_list[i] in class_list):
                    pet_list0 = list(filter(lambda x: class_list[x]=='0', range(len(class_list))))
                    pet_list1 = list(filter(lambda x: class_list[x]=='1', range(len(class_list))))
                    pet_list = pet_list0 + pet_list1
                    dngr_class = dngr_class + list(filter(lambda x: class_list[x]==dngr_list[i], range(len(class_list))))
            # print(pet_list)
            # print(dngr_class)
            for j in pet_list:
                x1 = last_file_content_splitList[(j*5)+1]
                y1 = last_file_content_splitList[(j*5)+2]
                print(x1, y1, end = 'pet\n')
                for k in dngr_class:
                    x2 = last_file_content_splitList[(k*5)+1]
                    y2 = last_file_content_splitList[(k*5)+2]
                    print(last_file_content_splitList[k*5], end = '\t')
                    print(x2, y2, end= 'dngr\n')

                    distance = math.sqrt((float(x2) - float(x1))**2 + (float(y2) - float(y1))**2)
                    distance *= 640
                    if(distance < 500):
                        print("경고! 위험물체가 가까이 있습니다!")
                        # oracle 서버 연결
                        connection = cx_Oracle.connect("dbdb", "dbdb", "project-db-stu.ddns.net:1524/xe")
                        cursor = connection.cursor()
                        
                        # sql 쿼리문 작성
                        
                        data = [(1, 1, cx_Oracle.DateFromTicks(datetime.now().timestamp()), dngr[last_file_content_splitList[k*5]])]
                        cursor.executemany("insert into records values (:1, :2, :3, :4)", data)
                        connection.commit()    
                        cursor.close()    
                        connection.close()