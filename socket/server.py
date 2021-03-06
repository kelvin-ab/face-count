import configparser
import tornado.web
import tornado.ioloop
import tornado.websocket as ws

import logging
import os

class SocketServer(ws.WebSocketHandler):

    @classmethod
    def route_urls(cls):
        return [(r'/', cls, {}), ]

    def open(self):
        print("New client connected")

    def on_message(self, message):
        self.write_message(message)

    def on_close(self):
        self.close()

    def check_origin(self, origin):
        return True


if __name__ == '__main__':
    logging.basicConfig(
        level=getattr(logging, os.getenv('LOGLEVEL', '').upper(), 'INFO'),
        format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    )
    config = configparser.ConfigParser()
    config.read('config.ini')
    soc_host = config['SOCKET']['HOST']
    soc_port = int(config['SOCKET']['PORT'])
    app = tornado.web.Application(SocketServer.route_urls(), websocket_max_message_size=50000000)
    app.listen(soc_port)
    tornado.ioloop.IOLoop.instance().start()
