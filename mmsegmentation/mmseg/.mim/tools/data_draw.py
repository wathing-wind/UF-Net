from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from PIL import Image, ImageDraw
import numpy as np

# driver_100_30frame/05251606_0445.MP4/03630.jpg
# driver_100_30frame/05251548_0439.MP4/00450.jpg
# driver_100_30frame/05251517_0433.MP4/01650.lines.txt'

image_dir = '/ai/volume/mmsegmentation/data/CULane/driver_100_30frame/05251606_0445.MP4/03630.jpg'
anno_dir = '/ai/volume/mmsegmentation/data/CULane/driver_100_30frame/05251606_0445.MP4/03630.lines.txt'
detect_dir = '/ai/volume/mmsegmentation/work_dirs/culane_xt/20240104_094441/culane_eval_tmp/driver_100_30frame/05251606_0445.MP4/03630.lines.txt'
detect_dir3 = '/ai/volume/Ultra-Fast-Lane-Detection-V2/tmp/culane_eval_tmp/driver_100_30frame/05251606_0445.MP4/03630.lines.txt'

x0, y0 = [], []
x1, y1 = [], []
x2, y2 = [], []
x3, y3 = [], []

with open(anno_dir) as anno_txt:
    lines1 = anno_txt.readlines()

with open(detect_dir) as detect_txt:
    lines2 = detect_txt.readlines()
with open(detect_dir3) as detect_txt:
    lines3 = detect_txt.readlines()

def cal_line(lines):

    for i, line in enumerate(lines):
        
        line = line.split('\n')
        
        line = line[0]

        values = [float(s) for s in line.split()]
        for j in range(0, len(values), 2):
            if i == 0:
                x0.append(values[j])
                y0.append(values[j+1])
            elif i == 1:
                x1.append(values[j])
                y1.append(values[j+1])   
            elif i == 2:
                x2.append(values[j])
                y2.append(values[j+1])
            elif i == 3:
                x3.append(values[j])
                y3.append(values[j+1]) 


def draw_line(lines):

    plt.plot(x0, y0, color='red')
    plt.plot(x1, y1, color='green')
    plt.plot(x2, y2, color='blue')
    plt.plot(x3, y3, color='yellow')
    plt.legend(labels=['1','2','3','4'],loc='best')

def draw_image(lines):

    for i, line in enumerate(lines):
        
        line = line.split('\n')
        
        line = line[0]

        values = [float(s) for s in line.split()]
        for j in range(0, len(values), 2):
            if i == 0:
                draw.ellipse((values[j]-2, values[j+1]-2, values[j]+2, values[j+1]+2), fill='red')
            elif i == 1:
                draw.ellipse((values[j]-2, values[j+1]-2, values[j]+2, values[j+1]+2), fill='green')
            elif i == 2:
                draw.ellipse((values[j]-2, values[j+1]-2, values[j]+2, values[j+1]+2), fill='blue')
            elif i == 3:
                draw.ellipse((values[j]-2, values[j+1]-2, values[j]+2, values[j+1]+2), fill='yellow')


if __name__ == "__main__":
    
    fig = plt.figure(1)
    image = Image.open(image_dir)
    draw = ImageDraw.Draw(image)
    # ground truth
    cal_line(lines1)
    draw_line(lines1)
    plt.savefig('plot1.png')
    draw_image(lines1)
    image.save('output_image1.jpg')
    x0, x1 ,x2, x3 = [], [] ,[], []
    y0 ,y1, y2, y3 = [], [] ,[], []

    # predict  
    fig = plt.figure(2)
    image = Image.open(image_dir)
    draw = ImageDraw.Draw(image)
    cal_line(lines2)
    draw_line(lines2)
    plt.savefig('plot2.png')
    draw_image(lines2)
    image.save('output_image2.jpg')
    x0, x1 ,x2, x3 = [], [] ,[], []
    y0 ,y1, y2, y3 = [], [] ,[], []

    fig = plt.figure(3)
    image = Image.open(image_dir)
    draw = ImageDraw.Draw(image)
    cal_line(lines3)
    draw_line(lines3)
    plt.savefig('plot3.png')
    draw_image(lines3)
    image.save('output_image3.jpg')

    # plt.show()
