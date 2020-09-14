import os

root = '/raid/krb/oxford-radar-robotcar-dataset/'
directories = os.listdir(root)

for dir in directories:
	if dir.split('-')[0] != '2019':
		continue
	os.system('./workspace/Documents/catkin_ws/build/test_feature_matching {} {}'.format(dir, dir))