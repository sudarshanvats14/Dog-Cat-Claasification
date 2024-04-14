from zipfile import ZipFile
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_path = 'archive (4).zip'

with ZipFile(data_path, 'r') as zip: 
	zip.extractall() 
	print('The data set has been extracted.') 


path = 'Data'
classes = os.listdir(path) 
print(classes)

fig = plt.gcf() 
fig.set_size_inches(16, 16) 

cat_dir = os.path.join('Data\\train\\cats') 
dog_dir = os.path.join('Data\\train\\dogs') 
cat_names = os.listdir(cat_dir) 
dog_names = os.listdir(dog_dir) 

pic_index = 210

cat_images = [os.path.join(cat_dir, fname) 
			for fname in cat_names[pic_index-8:pic_index]] 
dog_images = [os.path.join(dog_dir, fname) 
			for fname in dog_names[pic_index-8:pic_index]] 

for i, img_path in enumerate(cat_images + dog_images): 
	sp = plt.subplot(4, 4, i+1) 
	sp.axis('Off') 

	img = mpimg.imread(img_path) 
	plt.imshow(img) 

plt.show() 
