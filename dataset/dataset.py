import os

fg_net_img_path ='your/path'
fg_net_txt_path = 'your/path'

list = os.listdir(fg_net_img_path)

with open(fg_net_txt_path, 'w') as f:
    for img in list:
        id, age = img.split('.')[0].split('A')
        age = "".join(filter(str.isdigit, age))
        f.write(id + " images/" + img + " " + age + "\n")
