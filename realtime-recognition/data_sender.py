from redis import Redis 
import pickle 


host_address = '147.46.219.146' 
redis = Redis(host_address)

def dump(tuple):
    return pickle.dumps(tuple, protocol=2)

def send(key, item):
    redis.rpush(key, dump(item))
