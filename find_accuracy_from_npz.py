import numpy as np
from sklearn.metrics import confusion_matrix
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description= 'Finding accuracy using saved npz file')
  parser.add_argument('--npz_file_name', type=str)
  parser.add_argument('--npz_file_name2', type=str, default='')
  parser.add_argument('--fuse_two_files', type=str, default='F')
  args = parser.parse_args()

  if args.fuse_two_files == 'F': 
    data = np.load(args.npz_file_name) 
    score = data['scores']
    label = data['labels']
    video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in score]
  else:
    data1 = np.load(args.npz_file_name)
    data2 = np.load(args.npz_file_name2)

    score1 = data1['scores']
    score2 = data2['scores']
    label = data1['labels']
    
    score = (score1*2 + score2*3)/5
    
    video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in score]
  
  cf = confusion_matrix(label, video_pred).astype(float)
  cls_cnt = cf.sum(axis=1)
  cls_hit = np.diag(cf)
  cls_acc = cls_hit / cls_cnt
  print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    
