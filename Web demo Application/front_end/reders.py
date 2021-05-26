from flask import Flask, render_template, jsonify,request
import socket
import os

app = Flask(__name__)




@app.route('/')
def reders():
    return render_template('s.html')

@app.route('/test/',methods=['POST','GET'])
def order_get():
    data = conn.recv(16)
    print('\n',"recv:", data.decode("utf-8"),'\n')
    order_=int(data.decode("utf-8"))
    return jsonify({'order':order_})

if __name__ == '__main__':
    server = socket.socket()
    server.bind(('localhost', 12345))  # 绑定要监听端口=(服务器的ip地址+任意一个端口)
    server.listen(5)  # 监听

    print("监听连接请求")


    conn, addr = server.accept()
    print("收到来自{}请求".format(addr))

    app.run(debug = True,use_reloader=False)