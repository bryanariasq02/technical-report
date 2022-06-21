import getpass
import pymongo

URI = getpass.getpass()

client  = pymongo.MongoClient(URI)
db = client['3Bios']
GrupLAC = db['GrupLAC']
SiB = 'SiB'

query1 = {'categoria_gruplac':'articulo'}
docs = db.GrupLAC.find(query1).explain()['stages']
GrupLAC.create_index([("categoria_gruplac",1)])
docs2 = db.GrupLAC.find(query1).explain()['stages']
