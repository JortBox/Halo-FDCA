import pickle
import numpy
import json

#with open('file.pkl', 'rb') as file:
#    info = pickle.load(file)

def scan_dict(obj):
    print("-----BEGINNING SCAN OF DICTIONARY-----")
    for k,v in obj.items():
        if isinstance(v, (numpy.int64, numpy.float64)):
            print(k, type(v), "-> numpy int or float 64 in the dictionary")
        elif isinstance(v, list):
            print(k, type(v))
            for i, item in enumerate(v):
                if isinstance(item,(numpy.int64, numpy.float64)):
                    print(i,type(item),"-> numpy int or float 64 in the list" ) 
        else :
            print(k, type(v))
                
#scan_dict(info)

def conv(obj):
    print("-----BEGINNING CONVERSION OF FLOAT64 AND INT64 INTO FLOAT AND INT IN A DICTIONARY-----")
    for k,v in obj.items():
        if isinstance(v, numpy.int64):
            obj[k]=int(obj[k])
            print("...converting ",k ," from int64 to int...")
        elif isinstance(v, numpy.float64):
            obj[k]=float(obj[k])
            print("...converting ",k ," from float64 to float...")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, numpy.int64):
                    v[i]=int(item)
                    print("...converting ",i, type(v) ,k," list elemet from int64 to int...")
                elif isinstance(item, numpy.float64):
                    print("...converting ",i, type(v) ,k," list elemet from float64 to float...")
                    v[i]=float(item)
                    
#conv(info)
#scan_dict(info)
#with open(("test.json"), "w") as f:
#json.dump(info, f, indent=4)