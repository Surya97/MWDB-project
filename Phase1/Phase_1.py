import sys

arguments = sys.argv[1:]

task = arguments[0]

model = arguments[1]

if task == '1':
    image_id = arguments[2]
elif task == '2':
    folder_path = arguments[2]

if task == '3':
    image_id = arguments[2]
    k = arguments[3]

