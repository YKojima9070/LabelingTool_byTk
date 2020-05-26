import cv2
import glob
import os
import time
import numpy as np
import json
import copy
import tkinter
from tkinter import messagebox
import tkinter.filedialog as tkdialog
from PIL import Image, ImageTk


class App():
    def __init__(self, root, root_title):
    ###旧Opencv版
        self.drawing_flag = False
        self.ix, self.iy = -1, -1
        self.label_img = []
        self.label_data = []
        self.pts = []
        self.class_color_dict = {}
        self.draw_mode = ''

        self.cur_class = ''
        self.cur_img = []
        self.class_color = '#FF0000'
        self.make_class = 5
        self.stroke_width = 10
        self.trans = 50

        self.img_loop_trg = True
        self.label_loop_trg = True
        self.img_scale = 0
        self.img_window = [640, 480]

        self.shift = 1
        self.delta_sx, self.delta_sy = -1, -1
        self.org_sx, self.org_sy = -1, -1
        self.cur_sx, self.cur_sy = -1, -1
        self.moving_flag = False
        self.dst_img = []
        self.org_img_size = []
        
        self.dst_img = []

        ###Tk版追加
        self.root = root
        self.root.geometry("1680x1050")
        self.root.title(root_title)
        self.cur_img_num = 0


        self.label_dict = {"name":"TEST", "time":"2020-03-25T09:37:28.746Z", "version":"1.1.3",
                           "data":[]}

        #sg.theme('SystemDefault')

        
        ##GUIレイアウト
        self.label_button1 = tkinter.Button(root, text="PolyLine")
        self.label_button1.place(x=1500, y=10, width=150, height=50)

        self.label_button2 = tkinter.Button(root, text="Ellipse")
        self.label_button2.place(x=1500, y=70, width=150, height=50)

        self.label_button3 = tkinter.Button(root, text="Rectangle")
        self.label_button3.place(x=1500, y=130, width=150, height=50)

        self.label_button4 = tkinter.Button(root, text="Polygon")
        self.label_button4.place(x=1500, y=190, width=150, height=50)


        self.img_canvas = tkinter.Canvas(root, width=1400, height=1000)
        self.label_canvas = tkinter.Canvas(root, width=1400, height=1000)


    ##参照フォルダ指定
    def img_dir_get(self):
        img_format = [".png", ".jpg", ".jpeg", ".bmp"]
        tar_dir = os.path.dirname(tkdialog.askopenfilename(filetypes=[('all files', '*.*')],initialdir=os.getcwd()))

        self.img_list = [p for p in glob.glob("{0}/**".format(tar_dir), recursive=True) if os.path.splitext(p)[1] in img_format]

        [self.label_dict['data'].append({"fileName":os.path.basename(self.img_list[i]),
                                         "set":"",
                                         "classLabel":"",
                                         "regionLabel":[]}) 
        for i in range(len(self.img_list))]




    def img_get(self):
        img = cv2.imread(self.img_list[self.cur_img_num])
        
        self.org_img_size = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(1400, 1000))
            #img = Image.fromarray(img)
        self.org_img = Image.fromarray(img)
            
            #self.org_img = ImageTk.PhotoImage(img)
            #self.img_canvas.create_image(0, 0, image=self.org_img, anchor='nw')
            #self.img_canvas.place(x=0, y=0)

    def label_update(self):
        blank_img = np.zeros((self.org_img_size[0], self.org_img_size[1], 4)).astype(np.uint8)
        cv2.rectangle(blank_img, (200, 200), (500, 700), (0, 0, 0, 128), -1)
        print(blank_img.shape)

        blank_img[0:100, 100:1000]

        label_img = cv2.resize(blank_img, dsize=(1400, 1000))
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGRA2RGBA)
        label_img = Image.fromarray(label_img, 'RGBA')

        self.org_img.paste(label_img, mask=label_img)

        self.label_img = ImageTk.PhotoImage(self.org_img)
        self.img_canvas.create_image(0, 0, image=self.label_img, anchor='nw')
        self.img_canvas.place(x=0, y=0)

    def callback(self, event):
        if event.keysym == 'Right':
            self.cur_img_num += 1

        if event.keysym == 'Left':
            self.cur_img_num -= 1
    
    def main_loop(self):
        self.img_canvas.bind('<Right>', self.callback)
        self.img_canvas.bind('<Left>', self.callback)
        self.img_canvas.focus_set()

        self.img_get()
        self.label_update()

        self.root.mainloop()

app_class = App(tkinter.Tk(), 'TEST')
app_class.main_loop()

        ##Macではthreadingは動かない
        #threading.Thread(target=self.control_box2, args=[window]).start()
        #Process(target=self.control_box2).start()
        #self.img_cap(img_list)
"""
    ##############################################
        i = 0
        cv2.namedWindow('ImageWindow') 

        while self.img_loop_trg:
            self.label_data = []

            img_array = np.fromfile(img_list[i], dtype=np.uint8)
            org_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)            
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)
            
            self.org_img_size = org_img.shape
 
            for num_data in range(len(self.label_dict['data'])):
                if self.label_dict['data'][num_data]['fileName'] == os.path.basename(img_list[i]):
                    self.label_data = self.label_dict['data'][num_data]['regionLabel']
 
            cv2.setMouseCallback('ImageWindow', self.mouse_event)
            
            blank_img = np.zeros((self.org_img_size[0], self.org_img_size[1] ,4)).astype(np.uint8)            

            ## Showing process
            while self.label_loop_trg: 
                sta_time = time.perf_counter()

                event, values = window.read(timeout = 0)
                self.control_box(event, values)

                # Make blank data for label_data
                self.label_img = blank_img.copy()

                # Update label_data to blank data 
                self.label_update(self.label_data)


                # Marge image and label
                dst_img = self.img_overlay(org_img)


                end_time = time.perf_counter()

                # Zoom shift process
                #dst_img = self.affine_img(dst_img)


                # Resize image 
                dst_img = self.scale_box(dst_img, self.img_window[1], self.img_window[0])

                
                cv2.imshow('ImageWindow', dst_img)
                
            
                k = cv2.waitKey(1) & 0xFF
 
                if k == ord('r'):
                    i += 1
                    if i > len(img_list)-1:
                        i = 0
                    
                    break
 
                elif k == ord('b'):
                    i -= 1
                    if i < -len(img_list)+1:
                        i = 0
                    break

                print(end_time-sta_time)

        window.close()            
        cv2.destroyAllWindows()

    ###############################################
    

    ### Module ####
    def control_box(self, event, values):
        if event == sg.TIMEOUT_KEY:
            self.stroke_width = int(values['strokewidth'])
            self.trans = int(values['Trans'])

            for i in range(self.make_class):
                    
                _color = 'class{}_color'.format(str(i))

                try:
                    window.FindElement(_color).Update(_color, button_color=(values[_color],values[_color]))
                except:
                    None

                if values[_color]:
                    self.class_color_dict['class{}'.format(str(i))] = values[_color]

                else:
                    self.class_color_dict['class{}'.format(str(i))] = self.class_color

                if values['class{}'.format(str(i))]:
                    self.cur_class = 'class{}'.format(str(i))

        if event == 'PolyLine':
            self.draw_mode = 'PolyLine'
 
        elif event == 'Ellipse':
            self.draw_mode = 'Ellipse'
 
        elif event == 'Rectangle':
            self.draw_mode = 'Rectangle'
 
        elif event == 'Polygon':
            self.draw_mode = 'Polygon'
 
        elif event == 'SaveLabel':
            self.save_process(values)

        if event == 'Exit':
            self.label_loop_trg = False
            self.img_loop_trg = False


    ### -MainProcess- ###
    def img_cap(self, img_list):
        i = 0
        cv2.namedWindow('ImageWindow') 
        while self.img_loop_trg:
            self.label_data = []

            img_array = np.fromfile(img_list[i], dtype=np.uint8)
            org_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)            
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)
            
            self.org_img_size = org_img.shape
 
            for num_data in range(len(self.label_dict['data'])):
                if self.label_dict['data'][num_data]['fileName'] == os.path.basename(img_list[i]):
                    self.label_data = self.label_dict['data'][num_data]['regionLabel']
 
            cv2.setMouseCallback('ImageWindow', self.mouse_event)
            
            blank_img = np.zeros((self.org_img_size[0], self.org_img_size[1] ,4)).astype(np.uint8)            

            ## Showing process
            while self.label_loop_trg: 

                sta_time = time.perf_counter()

                # Make blank data for label_data
                self.label_img = blank_img.copy()

                # Update label_data to blank data 
                self.label_update(self.label_data)

                # Marge image and label
                dst_img = self.img_overlay(org_img)

                # Zoom shift process
                dst_img = self.affine_img(dst_img)
             
                end_time = time.perf_counter()

                # Resize image 
                dst_img = self.scale_box(dst_img, self.img_window[1], self.img_window[0])

                #print(end_time-sta_time)

                cv2.imshow('ImageWindow', dst_img)
            
                k = cv2.waitKey(1) & 0xFF
 
                if k == ord('r'):
                    i += 1
                    if i > len(img_list)-1:
                        i = 0
                    
                    break
 
                elif k == ord('b'):
                    i -= 1
                    if i < -len(img_list)+1:
                        i = 0
                    break
        
        cv2.destroyAllWindows()

 
    ## Reseize Image process
    def scale_box(self, img, width, height):
        self.img_scale = max(width / img.shape[1], height / img.shape[0])
        return cv2.resize(img, dsize=None, fx=self.img_scale, fy=self.img_scale)

 
    ## Updating each label data
    def label_update(self, label_data):

        for i in range(len(label_data)):

            self.num_img = i
            label = label_data[self.num_img]

            try:
                value = self.class_color_dict[label["className"]].lstrip('#')
                lv = len(value)
                color =  [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
                color.reverse()

            except:
                print('クラスが選択されていません')
                continue


            if label['type'] == "PolyLine":
                pts = np.array(label["points"], dtype=np.int32)
                cv2.polylines(self.label_img, [pts], False,  color, thickness=label["strokeWidth"])
                
            elif label['type'] == "Ellipse":
                cv2.ellipse(self.label_img, (label['x'],label['y']),
                            (label["radiusX"],label["radiusY"]),
                            angle = 0,
                            startAngle = 0,
                            endAngle = 360,
                            color = color,
                            thickness = -1)
 
            elif label['type'] == "Rect":
                cv2.rectangle(self.label_img, (label['x'],label['y']),
                             (label['width'],label['height']), color, -1)
 
            elif label['type'] == "PolyGon":
                pts = np.array(label["points"], dtype=np.int32)
                cv2.fillConvexPoly(self.label_img, pts, color)

    
    ## Make marged image
    def img_overlay(self, org_img):

        alpha = self.trans / 100

        self.label_img = cv2.addWeighted(org_img, alpha, self.label_img, 1 - alpha, 0)

        label_img_gray = cv2.cvtColor(self.label_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(label_img_gray, 255 * alpha + 5, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)
        org_img_bg = cv2.bitwise_and(org_img, org_img, mask=mask_inv)
        label_img_fg = cv2.bitwise_and(self.label_img, self.label_img, mask = mask)

        return cv2.add(org_img_bg, label_img_fg)

                
    ## Affine, shift, zoom
    def affine_img(self, img):
        
        ## Shift Image
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()
        dest[:,0] += self.cur_sx + self.delta_sx 
        dest[:,1] += self.cur_sy + self.delta_sy
        affine = cv2.getAffineTransform(src, dest)
        
        dst_img = cv2.warpAffine(img, affine, (self.org_img_size[1], self.org_img_size[0]))


        ## Zoom Image
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()

        dest[:,0] += - (self.org_img_size[0] / 2)
        dest[:,1] += - (self.org_img_size[1] / 2)

        dest = dest * self.shift

        dest[:,0] += (self.org_img_size[0] / 2)
        dest[:,1] += (self.org_img_size[1] / 2)

        affine = cv2.getAffineTransform(src, dest)                
        dst_img = cv2.warpAffine(dst_img, affine, (self.org_img_size[1], self.org_img_size[0]))
 
        return dst_img

    def pic_test(self, org_img):
        self.label_update(self.label_data)

        # Marge image and label
        dst_img = self.img_overlay(org_img)

        # Zoom shift process
        dst_img = self.affine_img(dst_img)

        # Resize image 
        self.dst_img = self.scale_box(dst_img, self.img_window[1], self.img_window[0])


##マウスイベント管理
    def mouse_event(self, event, x, y, flags, param):
        x = int((1 / self.img_scale) * x)
        y = int((1 / self.img_scale) * y)

        if not flags & cv2.EVENT_FLAG_SHIFTKEY:
            x = int((1/self.shift) * x - self.cur_sx + (self.org_img_size[0]-(self.org_img_size[0]/self.shift)) / 2)
            y = int((1/self.shift) * y - self.cur_sy + (self.org_img_size[1]-(self.org_img_size[1]/self.shift)) / 2)

            if self.draw_mode == "PolyLine":
                self.draw_polyline(event, x, y, flags, param)

            elif self.draw_mode == "Ellipse":
                self.draw_ellipse(event, x, y, flags, param)

            elif self.draw_mode == "Rectangle":
                self.draw_rectangle(event, x, y, flags, param)
 
            elif self.draw_mode == "Polygon":
                self.draw_polygon(event, x, y, flags, param)

            if event == cv2.EVENT_RBUTTONDOWN:
                if self.drawing_flag == False:
                    del self.label_data[-1]

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing_flag = False
        
        else:
            if event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    self.shift += 0.1

                else:
                    self.shift -= 0.1
                    if self.shift <= 1:
                        self.shift = 1
                
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.moving_flag = True
                self.org_sx, self.org_sy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.moving_flag:
                    self.delta_sx = x - self.org_sx
                    self.delta_sy = y - self.org_sy
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.moving_flag = False
                self.cur_sx  += self.delta_sx
                self.cur_sy += self.delta_sy
            
                self.delta_sx, self.delta_sy = 0, 0
        

##個別描画処理
    ##Polylineマウスイベント
    def draw_polyline(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
 
            self.drawing_flag = True
            self.pts = []
            self.pts.append([x, y])
            self.label_data.append({"className":self.cur_class,
                                    "type":"PolyLine",
                                    "strokeWidth":self.stroke_width,
                                    "points":[self.pts]})
 
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_flag == True:
                self.pts.append([x, y])
                self.label_data[-1]["points"] = self.pts
 
    ##Ellipseマウスイベント 
    def draw_ellipse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_flag = True
            self.ix, self.iy = x, y
            self.label_data.append({"className":self.cur_class,
                                    "type":"Ellipse",
                                    "x":self.ix,
                                    "y":self.iy,
                                    "radiusX":0,
                                    "radiusY":0})
             
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_flag == True:
                self.label_data[-1]["radiusX"] = abs(self.ix - x)
                self.label_data[-1]["radiusY"] = abs(self.iy - y)

    ##Rectangleマウスイベント                 
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_flag = True
            self.ix, self.iy = x, y
            self.label_data.append({"className":self.cur_class,
                                    "type":"Rect",
                                    "x":self.ix,
                                    "y":self.iy,
                                    "width":x,
                                    "height":y})
 
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_flag == True:
                self.label_data[-1]["width"] = x
                self.label_data[-1]["height"] = y            


    ##Polygonマウスイベント                  
    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
 
            self.drawing_flag = True
            self.pts = []
            self.pts.append([x, y])
            self.label_data.append({"className":self.cur_class,
                                    "type":"PolyGon",
                                    "points":[self.pts]})
 
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_flag == True:
                self.pts.append([x, y])
                self.label_data[-1]["points"] = self.pts
  
    def save_process(self, values):
        _class_name_update = {}
        for i in range(self.make_class):
            _class_name_update['class{}'.format(i)] = values['class_name{}'.format(i)]

            save_label_dict = copy.deepcopy(self.label_dict) ## Atten: dict is mutable data, do not use copy.copy() for image

        _sd = save_label_dict['data']

        for i in range(len(_sd)):
            for n in range(len(_sd[i]['regionLabel'])):
                _nv = _sd[i]['regionLabel'][n]
   
                _nv['className'] = _class_name_update[_nv['className']]  

                if _nv['type'] == 'Rect':
                    _nv['width'] = _nv['width'] - _nv['x'] 
                    _nv['height'] = _nv['height'] - _nv['y'] 

                 
        try:
            save_file = sg.popup_get_file('ラベル保存',
                                         file_types=(('JSONファイル', '*.json'),), 
                                         save_as = True)+'{}'.format('.json')

            with open(save_file, 'w') as f:
                json.dump(save_label_dict, f, ensure_ascii=False)
 
        except:
            print('保存先が指定されていません。')
    """                           

App(tkinter.Tk(), 'TEST')
 

