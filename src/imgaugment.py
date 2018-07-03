import Augmentor
p = Augmentor.Pipeline("/Users/raimibinkarim/Desktop/Original_Images_v3")

p.ground_truth("/Users/raimibinkarim/Desktop/Masks_Seg_v3")

p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.sample(2)
print("\n")
