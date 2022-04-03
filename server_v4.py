# -*- coding:utf-8 -*-
import socket
import cv2
import numpy as np
#import Image
import SocketServer
import sys
import datetime
#import commands
import os
#from pylsd.lsd import lsd

#######################引数でエッジ検出のtypeを分けれるようにする
#######################GoPro用で歪みをとっている
#print ("Waiting for connections...")

host = "192.168.11.12" #お使いのサーバーのホスト名を入れます
port = 12345 #クライアントと同じPORTをしてあげます


#serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#serversock.bind((host,port)) #IPとPORTを指定してバインドします
#serversock.listen(10) #接続の待ち受けをします（キューの最大数を指定）
#lsd画像に変換

global wide

global height

wide = 480
height = 270


def mylsd (a):
    #src = cv2.imread('undistort/und_%s' % (filename),cv2.IMREAD_COLOR)
    #print ("Input an image file with the path ( ex. test/XXXX.yyy):")
    #infile = input(a)
    #src = cv2.imread(infile)
    src = a


    RGB = cv2.split(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray)
    #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #lines = lsd(gray)
    linesB = lsd(RGB[0])
    linesG = lsd(RGB[1])
    linesR = lsd(RGB[2])

    size = src.size

    white_img = np.zeros(size, dtype=np.uint8)
    #white_img.fill(255)
    limg = src

    limg.fill(0)

    for i in range(linesB.shape[0]):
        pt1 = (int(linesB[i,0]), int(linesB[i,1]))
        pt2 = (int(linesB[i,2]), int(linesB[i,3]))
        width = linesB[i,4]
        cv2.line(limg, pt1, pt2, (255,255,255), int(np.ceil(width / 2)))

    for i in range(linesG.shape[0]):
        pt1 = (int(linesG[i,0]), int(linesG[i,1]))
        pt2 = (int(linesG[i,2]), int(linesG[i,3]))
        width = linesG[i,4]
        cv2.line(limg, pt1, pt2, (255,255,255), int(np.ceil(width / 2)))

    for i in range(linesR.shape[0]):
        pt1 = (int(linesR[i,0]), int(linesR[i,1]))
        pt2 = (int(linesR[i,2]), int(linesR[i,3]))
        width = linesR[i,4]
        cv2.line(limg, pt1, pt2, (255,255,255), int(np.ceil(width / 2)))


    #limg = cv2.rectangle(limg,(0,0),(328,246),(0,0,0),3)

    #cv2.imshow('src',src)
    #cv2.imshow('limg',limg)
    #cv2.waitKey(5000)
    return (limg)

def rotation (src,angle):
    # 画像読み込み
    #img_src = cv2.imread("/home/pi/Desktop/image.png")
    img_src = src

    # 画像サイズの取得(横, 縦)
    #print(img_src.shape[0])
    size = tuple([img_src.shape[1], img_src.shape[0]])
    h, w = img_src.shape[:2]

    # 画像の中心位置(x, y)
    center = tuple([int(size[0]/2), int(size[1]/2)])
    # 拡大比率
    scale = 1.0

    angle = float(angle)
    angle = -1*angle


    # 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # アフィン変換
    img_rot = cv2.warpAffine(img_src, rotation_matrix, size, flags=cv2.INTER_CUBIC)

    print(img_rot.shape)



    return(img_rot)

def communicate1(clientsock, client_address):
    rcvmsg = client.recv(1024)
    print(rcvmsg)
    s_msg = "HELLO"
    clientsock.sendall(s_msg)

    #画像のbyte数を受け取る
    size = self.request.recv(1024).encode('ascii')
    size = int(size)
    size = size
    return_msg = "receive_size"
    print("確認")
    print(size)
    self.request.send(return_msg.encode()) #メッセージを返します

    #画像の受け取り
    buf=b''
    recvlen=100
    while recvlen>0:
        receivedstr=self.request.recv(62208)
        #receivedstr=str(receivedstr,'utf-8')
        recvlen=len(receivedstr)
        #if receivedstr in "finish": break
        #print (len(receivedstr))
        buf +=receivedstr
        print(sys.getsizeof(buf))
        if (sys.getsizeof(buf) == size):
            break
    #receivedstr=str(receivedstr,'utf-8')
    receivedstr=str(receivedstr)
    print ("check2")
    #receivedstr=receivedstr.tostring()
    narray=np.fromstring(buf,dtype='uint8')
    print ("check2")
    narray = cv2.imdecode(narray,1)
    print ("check3")
    #時間の取得
    date = datetime.datetime.today()
    print ("check4")

    #センシング画像を表示
    cv2.imshow('img',narray)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    PATH ='D:\\dev\\workspace\\UMap\\SEN\\PiCA\\1.img\\'+ rcvmsg+'.jpg'
    cv2.imwrite(PATH,narray)

    clientsock.close()

    return(narray)

def communicate2():
    #result is send to Matching program
    LocalHOST = '127.0.0.1'    # The remote host
    LocalPORT = 20001          # The same port as used by the server

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((LocalHOST, LocalPORT))

    PATH ='D:\\dev\\workspace\\UMap\\SEN\\PiCA\\1.img\\'+ rcvmsg+'.jpg'

    #message = "%s/ %s.jpg dummy" % (folder, rcvmeg)
    message = PATH
    s.send(message)
    #location = s.recv(4096).encode('ascii')
    location = s.recv(4096)
    print(type(location))
    print(len(location))
    print(location)
    s.close

def communicate3():
    #result is send to Matching program
    Localhost = '192.168.11.17'    # The remote host
    Localport = 20001          # The same port as used by the server
    robot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robot.connect((Localhost, Localport))

    result = location
    robot.send(location)


def main():
    while True:
        print 'Waiting for connections...'
        host = "192.168.11.12" #お使いのサーバーのホスト名を入れます
        port = 1245 #クライアントで設定したPORTと同じもの指定してあげます

        serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serversock.bind((host,port)) #IPとPORTを指定してバインドします
        serversock.listen(10) #接続の待ち受けをします（キューの最大数を指定）

        clientsock, client_address = serversock.accept() #接続されればデータを格納

        narray = communicate1(clientsock, client_address)

        #線分画像に変換
        a = mylsd(narray)
        PATH ='D:\\dev\\workspace\\UMap\\SEN\\PiCA\\2.lsdimg\\'+ rcvmsg +'.jpg'
        cv2.imwrite(PATH,a)


        PATH ='D:\\dev\\workspace\\UMap\\SEN\\PiCA\\4.query\\'+ rcvmsg +'.png'
        cv2.imwrite(PATH,b)
        cv2.imshow ('lsd',a)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        communicate2()

        communicate3()

main()
SocketServer.TCPServer.allow_reuse_address = True
server = SocketServer.TCPServer((host, port), main)
#server = socketserver.ThreadingTCPServer((HOST, PORT), TCPHandler)
#^Cを押したときにソケットを閉じる
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
server.shutdown()
sys.exit()
