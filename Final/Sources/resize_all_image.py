import os
from PIL import Image
resize_method = Image.ANTIALIAS

max_height= 800
max_width= 800
extensions= ['JPG']

path= os.path.abspath("./monkey_data/")
save_path = os.path.abspath("./monkey_data_new/")

def adjusted_size(width,height):
    if width>max_width or height>max_height:
        if width>height:
            return max_width, int (max_width * height/ width)
        else:
            return int (max_height*width/height), max_height
    else:
        return width,height
	
if __name__ == "__main__":
    for f in os.listdir(path):
        #print(f)
        if os.path.isfile(os.path.join(path,f)):
            f_text, f_ext= os.path.splitext(f)
            f_ext = f_ext[1:].upper()
            if f_ext in extensions:
                print(f)
                image = Image.open(os.path.join(path,f))
                width, height= image.size
                image = image.resize(adjusted_size(width, height))
#                image = image.convert("RGBA")
                image.save(os.path.join(save_path,f))
