# CUDA_VISIBLE_DEVICES=0 python image_transfer.py \
# 	--mode photorealistic --ckpoint checkpoints/photo_image.pt \
# 	--auto_seg \
# 	--content data/content/01.jpg  \
# 	--style data/style/01.jpg


CUDA_VISIBLE_DEVICES=0 python image_transfer.py \
	--mode photorealistic --ckpoint checkpoints/photo_image.pt \
	--auto_seg \
	--content data/content/05.jpg  \
	--style data/style/05.jpg

# CUDA_VISIBLE_DEVICES=0 python image_transfer.py \
# 	--mode artistic --ckpoint checkpoints/art_image.pt \
# 	--auto_seg \
# 	--content data/content/02.jpg  \
# 	--style data/style/02.png