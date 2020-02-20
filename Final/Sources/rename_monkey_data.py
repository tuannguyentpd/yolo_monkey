import os

mydirec = "./monkey_data/"

def rename_multi_file(): 
    i = 0
      
    for filename in os.listdir(mydirec): 
        src = mydirec + filename
        dst = mydirec + "monkey_" + str(i) + ".jpg"
          
        os.rename(src, dst)
        print("rename file: %s -> %s"%(src, dst)) 
        i += 1
  

if __name__ == '__main__': 
      
    rename_multi_file()
