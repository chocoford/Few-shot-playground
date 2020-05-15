# VOC 1-way 1-shot
python test.py with gpu_id=0 mode='test' snapshot="./runs/fpn/PANet_VOC_align_sets_0_1way_1shot_[train]/$1/snapshots/$2.pth" theme='fpn'
python test.py with gpu_id=0 mode='test' snapshot="./runs/fpn/PANet_VOC_align_sets_1_1way_1shot_[train]/$1/snapshots/$2.pth" theme='fpn'
#python test.py with gpu_id=0 mode='test' snapshot='./runs/fpn/PANet_VOC_align_sets_2_1way_1shot_[train]/$1/snapshots/$2.pth' theme='fpn'
#python test.py with gpu_id=0 mode='test' snapshot='./runs/fpn/PANet_VOC_align_sets_3_1way_1shot_[train]/$1/snapshots/$2.pth' theme='fpn'