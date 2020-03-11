import socket
import json
import threading

class RlWorldClient():

    def __init__(self,ip_address,port_number):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip_address,port_number))
        self.sock.setblocking(False) #same as settimeout(0.0)
        self.observation_dict = {}

        self.recv_bytes_buffer = b""


    def read_observation_dict(self):

        #keep receiving bytes until its run out
        while True:
            #read bytes
            try:
                
                new_bytes = self.sock.recv(1024)
            except:
                break

            if len(new_bytes) == 0:
                break

            #append the new bytes to the buffer
            self.recv_bytes_buffer += new_bytes
           

        #split the list by the null character. json bytes should be delimited by the null character
        json_chuck_list = self.recv_bytes_buffer.split(b'\x00')

        #if there are at least two chuck eg 1 delimiting character
        if len(json_chuck_list) >= 2:

            #grab the second to last chuck as the latest json dict
            json_bytes = json_chuck_list[-2]

            #fill the buffer with the bytes left over after the last delimiting char
            self.recv_bytes_buffer = json_chuck_list[-1]

            #decode the bytes into a string
            json_str = json_bytes.decode("utf-8")

            #convert the json string to python dict object
            self.observation_dict  = json.loads(json_str)

        return self.observation_dict

        

    def send_action_dict(self,action_dict):
        
        json_str = json.dumps(action_dict)

        #encode the string into bytes and add the null character to the end of the string as a marker
        json_bytes = json_str.encode("utf-8") + b'\x00'

        self.sock.sendall(json_bytes)