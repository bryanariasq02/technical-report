import getpass
import pymongo

URI = getpass.getpass()

client  = pymongo.MongoClient(URI)
db = client['3Bios']
GrupLAC = 'GrupLAC'
SiB = 'SiB'

query1 = {'categoria_gruplac':'articulo'}

docs = db.GrupLAC.find(query1).explain()['stages']
