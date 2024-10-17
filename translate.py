from PIL import Image

# Open the image
image = Image.open("image_a.png")  # Replace "image.png" with your image file name

# Flip the image horizontally
translated_image = image.transpose(Image.FLIP_LEFT_RIGHT)

# Rotate the image 30 degrees clockwise while maintaining size
translated_image = translated_image.rotate(-30, expand=True, resample=Image.BICUBIC)

# Calculate the size difference and crop to maintain original size
width, height = image.size
rotated_width, rotated_height = translated_image.size
left = (rotated_width - width) / 2
top = (rotated_height - height) / 2
right = (rotated_width + width) / 2
bottom = (rotated_height + height) / 2

final_image = translated_image.crop((left, top, right, bottom))

# Save the result
final_image.save("image_b.png")

print("Image processing complete. Result saved as 'image_b.png'.")

