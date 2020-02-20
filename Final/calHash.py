import hashlib
'''
print("############# Cach 1 #############") 
#filename = input("Enter the input file name: ")
with open('1612774v2.zip',"rb") as f:
    bytes = f.read() # read entire file as bytes
    readable_hash = hashlib.sha256(bytes).hexdigest();
    print(readable_hash)


print("############# Cach 2 #############") 
#filename = input("Enter the input file name: ")
sha256_hash = hashlib.sha256()
with open('1612774v2.zip',"rb") as f:
    # Read and update hash string value in blocks of 4K
    for byte_block in iter(lambda: f.read(4096),b""):
        sha256_hash.update(byte_block)
    print(sha256_hash.hexdigest())


print("############# Cach 3 #############") 
BLOCKSIZE = 65536
sha = hashlib.sha256()
with open('1612774v2.zip', 'rb') as kali_file:
    file_buffer = kali_file.read(BLOCKSIZE)
    while len(file_buffer) > 0:
        sha.update(file_buffer)
        file_buffer = kali_file.read(BLOCKSIZE)       
    print (len(sha.hexdigest()))
'''

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

print(md5('1612774.zip'))
print("len = %d"%len(md5('1612774.zip')))


