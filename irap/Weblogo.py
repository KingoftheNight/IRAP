import os
import sys
import numpy as np
import math
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
import Visual as ivis

def weblogo_read(file):
    with open(file, 'r') as f:
        data = f.readlines()
    out = []
    for i in data:
        if '>' not in i:
            out.append(i.strip('\n'))
    return out

def weblogo_check(value):
    H = 0
    for i in value:
        if i != 0:
            H += i*math.log(i, 2)
    Rseq = math.log(len(value), 2) + H
    out = []
    for i in value:
        if i != 0:
            out.append(Rseq*i)
        else:
            out.append(0)
    return out

def weblogo_count(data, tp='p'):
    aa_index = ivis.visual_create_aa(tp=tp)
    count = []
    for i in range(len(data[0])):
        mid = ivis.visual_create_n_matrix(x=len(aa_index))
        for line in data:
            mid[aa_index.index(line[i])] += 1
        count.append(mid)
    count = np.array(count)
    percent = []
    for i in range(len(count)):
        mid = []
        for j in range(len(count[i])):
            mid.append(count[i,j]/np.sum(count[i]))
        mid = weblogo_check(mid)
        percent.append(mid)
    return percent

# create path index
def weblogo_path(tp='p'):
    if tp == 'n':
        path = {
            'A':'M60,100H48l-7-30.6H19L12,100H0L25,0h10L60,100z M38.5,58.4l-8-35.3h-1l-8,35.3H38.5z',
            'T':'M60,11.1H36.7V100H23.3V11.1H0V0h60V11.1z',
            'C':'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z',
            'G':'M60,100h-9.7l-1.7-8c-1.5,2.3-3.9,4.2-7.1,5.7c-3.2,1.5-6.8,2.3-10.6,2.3c-8,0-15.1-3.7-21.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,2.9-31.2,8.6-40C14.3,4.4,22.1,0,32,0c8.8,0,15.6,3.2,20.6,9.7c4.9,6.5,7.4,14.9,7.4,25.1H47.4		c0-8-1.3-14-4-18c-2.7-4-6.5-6-11.4-6c-6.1,0-10.7,3.1-13.7,9.4c-3.1,6.3-4.6,16.9-4.6,31.7c0,14.1,1.9,23.8,5.7,29.1		c3.8,5.3,8.2,8,13.1,8c4.9,0,8.9-1.8,11.7-5.4c2.9-3.6,4.3-9.2,4.3-16.9v-6.3H30.9V49.7H60V100z'
            }
    else:
        path = {
            'A':'M60,100H48l-7-30.6H19L12,100H0L25,0h10L60,100z M38.5,58.4l-8-35.3h-1l-8,35.3H38.5z',
            'R':'M60,100H46.8L27,56.7H12.7V100H0V0h27.5c8.8,0,16,2.2,21.5,6.7c5.5,4.5,8.3,11.8,8.3,21.9		c0,7.8-1.8,13.8-5.5,18.1c-3.7,4.3-7.9,7-12.7,8.2L60,100z M44.6,28.7c0-5.5-1.6-9.7-4.7-12.9c-3.1-3.1-8.2-4.7-15.1-4.7H12.7v34.5		h15.4c4.4,0,8.3-1.3,11.6-3.8C42.9,39.3,44.6,34.9,44.6,28.7z',
            'N':'M60,100H45.6L13.8,27.5h-0.6V100H0V0h14.4l31.7,72.5h0.6V0H60V100z',
            'D':'M60,50.3c0,17.9-3.7,30.7-11,38.3C41.6,96.2,31.7,100,19.2,100H0V0h19.2c14,0,24.2,3.9,30.8,11.7		C56.7,19.5,60,32.4,60,50.3z M46.4,50.3c0-14.4-2.3-24.6-6.8-30.4c-4.5-5.8-11.3-8.8-20.4-8.8H13v77.8h6.2c9.1,0,15.8-2.8,20.4-8.5		C44.2,74.8,46.4,64.7,46.4,50.3z',
            'C':'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z',
            'Q':'M60,46.8c0,8.6-0.6,15.9-1.9,21.8c-1.3,5.9-3.3,10.7-6.2,14.2l4.3,8.1l-8.6,9.1l-4.9-9.1		c-1.4,1.1-3.2,1.9-5.4,2.4c-2.2,0.5-4.5,0.8-7,0.8c-9.4,0-16.8-3.7-22.2-11C2.7,75.7,0,63.6,0,46.8c0-16.8,2.7-28.8,8.1-36		C13.5,3.6,20.9,0,30.3,0c9.4,0,16.7,3.6,21.9,10.8C57.4,17.9,60,29.9,60,46.8z M47,46.8c0-15.1-1.6-24.9-4.9-29.6		c-3.2-4.7-7.2-7-11.9-7c-4.7,0-8.7,2.3-12.2,7C14.7,21.9,13,31.7,13,46.8s1.7,25,5.1,29.8c3.4,4.8,7.5,7.3,12.2,7.3		c1.8,0,3.4-0.3,4.9-0.8c1.4-0.5,2.5-1.2,3.2-1.9l-8.6-16.7l7.6-9.1l7.6,14.5c0.7-3.6,1.3-6.8,1.6-9.7C46.8,57.4,47,52.9,47,46.8z',
            'E':'M60,100H0V0h57.1v11.1H13.5v31h40v11.1h-40v35.7H60V100z',
            'G':'M60,100h-9.7l-1.7-8c-1.5,2.3-3.9,4.2-7.1,5.7c-3.2,1.5-6.8,2.3-10.6,2.3c-8,0-15.1-3.7-21.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,2.9-31.2,8.6-40C14.3,4.4,22.1,0,32,0c8.8,0,15.6,3.2,20.6,9.7c4.9,6.5,7.4,14.9,7.4,25.1H47.4		c0-8-1.3-14-4-18c-2.7-4-6.5-6-11.4-6c-6.1,0-10.7,3.1-13.7,9.4c-3.1,6.3-4.6,16.9-4.6,31.7c0,14.1,1.9,23.8,5.7,29.1		c3.8,5.3,8.2,8,13.1,8c4.9,0,8.9-1.8,11.7-5.4c2.9-3.6,4.3-9.2,4.3-16.9v-6.3H30.9V49.7H60V100z',
            'H':'M60,100H46.6V53.2H13.4V100H0V0h13.4v42.1h33.2V0H60V100z',
            'I':'M23.1,11.5H0V0h60v11.5H36.9v76.9H60V100H0V88.4h23.1V11.5z',
            'L':'M60,100H0V0h13.3v88.9H60V100z',
            'K':'M60,100H45.9L21.1,50.3l-8.6,12.3V100H0V0h12.4v43.3L43.2,0h14.6L29.2,39.2L60,100z',
            'M':'M60,100H48.9V28.7h-1.1L34.4,100h-8.9L12.2,28.7h-1.1V100H0V0h17.2l12.2,66.1h1.1L42.8,0H60V100z',
            'F':'M60,11.1H13.1v32.7h35.4V55H13.1v45H0V0h60V11.1z',
            'P':'M60,30.4c0,9.4-2.7,16.8-8,22.2c-5.3,5.5-12.8,8.2-22.3,8.2H13.1V100H0V0h29.7C39.2,0,46.7,2.7,52,8.2		C57.3,13.7,60,21.1,60,30.4z M46.9,30.4c0-7.4-1.7-12.5-5.1-15.2c-3.4-2.7-8.6-4.1-15.4-4.1H13.1v38.6h13.1c6.9,0,12-1.4,15.4-4.1		C45.1,42.9,46.9,37.8,46.9,30.4z',
            'S':'M60,71.4c0,8.8-2.7,15.7-8.1,20.9c-5.4,5.1-12.8,7.7-22.1,7.7c-9.3,0-16.6-2.7-21.9-8C2.6,86.7,0,80,0,72v-4		h12.9v3.4c0,5.7,1.7,10.1,5,13.1c3.4,3.1,7.3,4.6,11.8,4.6c6,0,10.4-1.6,13.2-4.9c2.8-3.2,4.2-7.1,4.2-11.7c0-3.8-1.7-7.3-5-10.6		c-3.4-3.2-8.2-6.2-14.6-8.9c-9-3.4-15.4-7.2-19.3-11.4c-3.9-4.2-5.9-9.1-5.9-14.9c0-8,2.7-14.5,8.1-19.4C15.8,2.5,22.2,0,29.7,0		c9.7,0,16.7,3,21,8.9c4.3,5.9,6.4,12.3,6.4,19.1H44.3c0.4-4.2-0.8-8-3.4-11.4c-2.6-3.4-6.4-5.1-11.2-5.1c-4.5,0-8,1.2-10.7,3.7		c-2.6,2.5-3.9,5.8-3.9,10c0,3.4,1,6.4,3.1,8.9c2.1,2.5,7.4,5.4,16,8.9c8.2,3.4,14.6,7.5,19.1,12.3C57.8,59.9,60,65.3,60,71.4z',
            'T':'M60,11.1H36.7V100H23.3V11.1H0V0h60V11.1z',
            'W':'M60,0L49.2,100h-9.8l-8.9-74h-1l-8.9,74h-9.8L0,0h10.8l5.4,63h1l7.4-63h10.8l7.4,63h1l5.4-63H60z',
            'Y':'M60,0L35.9,55v45H24.1V55L0,0h11.8l17.9,42.7h0.5L48.2,0H60z',
            'V':'M60,0L35,100H25L0,0h12l17.5,75.7h1L48,0H60z'
            }
    return path

# create aa color
def weblogo_color(tp='p'):
    if tp != 'p':
        color = {
            'A':'#07d794',
            'T':'#786fa6',
            'C':'#e15f41',
            'G':'#3dc1d3',
            }
    else:
        color = {
            'A':'#f5cd79',
            'R':'#3dc1d3',
            'N':'#ea8685',
            'D':'#cf6a87',
            'C':'#778beb',
            'Q':'#f78fb3',
            'E':'#f3a683',
            'G':'#f7d794',
            'H':'#546de5',
            'I':'#e15f41',
            'L':'#f8a5c2',
            'K':'#786fa6',
            'M':'#63cdda',
            'F':'#f19066',
            'P':'#574b90',
            'S':'#e66767',
            'T':'#303952',
            'W':'#778beb',
            'Y':'#cf6a87',
            'V':'#07d794'
            }
    return color

# create weblogo ruler
def weblogo_ruler(number):
    out = '<rect x="45" y="0" width="2" height="110" fill="#333333"/><text fill="#333333" x="25" y="114">0</text><rect x="36" y="108" width="10" height="2" fill="#333333"/><rect x="41" y="103" width="5" height="2" fill="#333333"/><rect x="41" y="98" width="5" height="2" fill="#333333"/><rect x="41" y="93" width="5" height="2" fill="#333333"/><rect x="41" y="88" width="5" height="2" fill="#333333"/><text fill="#333333" x="25" y="89">1</text><rect x="36" y="83" width="10" height="2" fill="#333333"/><rect x="41" y="78" width="5" height="2" fill="#333333"/><rect x="41" y="73" width="5" height="2" fill="#333333"/><rect x="41" y="68" width="5" height="2" fill="#333333"/><rect x="41" y="63" width="5" height="2" fill="#333333"/><text fill="#333333" x="25" y="64">2</text><rect x="36" y="58" width="10" height="2" fill="#333333"/><rect x="41" y="53" width="5" height="2" fill="#333333"/><rect x="41" y="48" width="5" height="2" fill="#333333"/><rect x="41" y="43" width="5" height="2" fill="#333333"/><rect x="41" y="38" width="5" height="2" fill="#333333"/><text fill="#333333" x="25" y="39">3</text><rect x="36" y="33" width="10" height="2" fill="#333333"/><rect x="41" y="28" width="5" height="2" fill="#333333"/><rect x="41" y="23" width="5" height="2" fill="#333333"/><rect x="41" y="18" width="5" height="2" fill="#333333"/><rect x="41" y="13" width="5" height="2" fill="#333333"/><text fill="#333333" x="25" y="14">4</text><rect x="36" y="8" width="10" height="2" fill="#333333"/><rect x="41" y="3" width="5" height="2" fill="#333333"/><rect x="41" y="-2" width="5" height="2" fill="#333333"/><text fill="#333333" x="0" y="10" transform="translate(0,55)rotate(-90, 10, 5)">bits</text>'
    out += '<rect x="45" y="108" width="' + str(int(number*24+50)) + '" height="2" fill="#333333"/><rect x="57" y="109" width="2" height="5" fill="#333333"/>'
    x = 33
    for j in range(int(number/5)+1):
        for i in range(4):
            x += 24
            out += '<rect x="' + str(x) + '" y="109" width="2" height="5" fill="#333333"/>'
        x += 24
        out += '<text fill="#333333" x="' + str(x-4) + '" y="134">' + str((j+1)*5) + '</text><rect x="' + str(x) + '" y="109" width="2" height="10" fill="#333333"/>'
    return out

# delete zero
def weblogo_del(line, aa_index):
    out1 = []
    out2 = []
    for i in range(len(line)):
        if line[i] != 0:
            out1.append(line[i])
            out2.append(aa_index[i])
    return [out1, out2]

# weblogo update line
def weblogo_update(line, aa):
    out = [[],[]]
    test = 'yes'
    for i in range(len(line[0])):
        if aa != line[0][i]:
            out[0] = out[0] + [line[0][i]]
            out[1] = out[1] + [line[1][i]]
        elif test != 'yes':
            out[0] = out[0] + [line[0][i]]
            out[1] = out[1] + [line[1][i]]
        else:
            test = 'no'
    return out

# count y and scale
def weblogo_yc(line, y1, y2, r):
    aa_value = min(line[0])
    aa_id = line[1][line[0].index(aa_value)]
    new_line = weblogo_update(line, aa_value)
    r += aa_value
    y1 -= r*25
    c = (y2-y1) / 100
    return y1, y2, c, new_line, r, aa_id

# create weblogo logo
def weblogo_logo(value, aa_index, path_index, color):
    x = 23
    logo = ''
    for i in range(len(value)):
        x += 24
        y2 = 108
        r = 0
        line = weblogo_del(value[i], aa_index)
        li = len(line[0])
        for j in range(li):
            y1 = 108
            if len(line[0]) != 0:
                y1, y2, c, line, r, aa_id = weblogo_yc(line, y1, y2, r)
                y2 = y1
                logo += '<path fill="' + color[aa_id] + '" d="' + path_index[aa_id] + '" transform="translate(' + str(x) + ',' + str(y1) + ')scale(0.4,' + str(c) + ')"/>'
    return logo

# weblogo plot
def weblogo_plot(value, title='Weblogo', out=now_path, tp='p'):
    aa_index = ivis.visual_create_aa(tp=tp)
    path_index = weblogo_path(tp=tp)
    color = weblogo_color(tp=tp)
    head = '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(int(len(value)*24+100)) + '" height="135">'
    end = '</svg>'
    ruler = weblogo_ruler(len(value))
    logo = weblogo_logo(value, aa_index, path_index, color)
    all_data = head + ruler + logo + end
    with open(out, 'w') as f:
        f.write(all_data)
    
# weblogo main
def weblogo(file=None, data=None, out=now_path, tp='p'):
    time_id = ivis.visual_timestamp()
    if data == None:
        data = weblogo_read(file)
    else:
        data = data
    if out == now_path:
        out = os.path.join(out, time_id + '-.svg')
    value = weblogo_count(data, tp=tp)
    weblogo_plot(value, out=out, tp=tp)
