from smilegan3 import gen_img_with_vector_train_data, gen_emotion_row, gen_emotion
from vector import calculate_and_export_avg_vectors
from visualize import visualize_avg_vectors

# --------------------------------------------- Step 1. --------------------------------------------- #

# Generate images with their corresponding vectors for labeling
# Use an offset, in case you want to generate images in multiple runs
# Images will be exported to img/train/train_img
# Create path if it doesn't exist 

gen_img_with_vector_train_data(1000, 0)


# --------------------------------------------- Step 2. --------------------------------------------- #

# Label images in img/train/labels.csv
# Calculate average vectors for each emotion

calculate_and_export_avg_vectors()

# Optional: Update vectors vars in model.py using the output-file average_vectors.csv

# Optional: Visualize vectors

visualize_avg_vectors()


# --------------------------------------------- Step 3. --------------------------------------------- #

# Generate emotions, using one of the two functions

# Generate random image and 'paint' all emotions on the random image
gen_emotion_row('Emotion Row')

# Or, just create a single image
gen_emotion('happy', 'Happy')
