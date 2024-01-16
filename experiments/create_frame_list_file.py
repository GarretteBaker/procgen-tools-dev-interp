import os

directory = './frames'
frame_rate = 24  # Change this to your desired frame rate

with open('file_list.txt', 'w') as file:
    for filename in sorted(os.listdir(directory), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith('.png'):
            file.write(f"file '{directory}/{filename}'\nduration {1/frame_rate}\n")
