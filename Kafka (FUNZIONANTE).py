from time import sleep
from ctypes import CDLL
import json
import pandas as pd
CDLL(r'C:\Users\carlo\anaconda3\Lib\site-packages\confluent_kafka.libs\librdkafka-5d2e2910.dll')

from confluent_kafka import Consumer

consumer_conf = dict()
consumer_conf['bootstrap.servers'] = '10.48.134.234:9092'
consumer_conf['group.id'] = 'carlo6'  #ricordarsi di group id ogni volta
consumer_conf['auto.offset.reset'] = 'latest'
kafka_consumer = Consumer(consumer_conf)
kafka_consumer.subscribe(['opc2k_topic'])
messages=list()
for i in range(0,200):    #questo valore non conta i secondi giusti
    sample_message = kafka_consumer.poll(timeout=10.0)
    messages.append(sample_message)
    sleep(0.5)
# if sample_message is None:
#     print('No message received from consumer')
# elif not sample_message.error():
#     print('Message received:')
    print(sample_message.value())

for j in range(len(messages)):
    print(messages[j].value())
kafka_consumer.close()

a=messages[0].value()
user_dict=json.loads(a.decode('utf-8'))

mess=pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                       for i in user_dict.keys() 
                       for j in user_dict[i].keys()},
                    orient='index')

for j in range(len(messages)):
    a = messages[j].value()
    user_dict=json.loads(a.decode('utf-8'))

    mess[j]=pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                       for i in user_dict.keys() 
                       for j in user_dict[i].keys()},
                    orient='index')
    
mess.to_excel(r'C:\Users\carlo\Desktop\Smart manufacturing\i40 Lab\data1.xlsx')

