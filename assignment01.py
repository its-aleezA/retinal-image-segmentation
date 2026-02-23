import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# functions
# gets the location of the images
def get_drishti_paths(base_dir, mode="Testing"):
    data_pairs = []  # dictionary pairs (one fundus image and its two ground truth masks (disc and cup))
    subset_path = os.path.join(base_dir, mode)  # to make a path with the right folder (Traing/Testing)
    img_root = os.path.join(subset_path, "Images")  # folder for the actual fundus images
    gt_root = os.path.join(subset_path, "GT")  # SoftMap folder for the actual known outputs
    for category in ["GLAUCOMA", "NORMAL"]:  # goes through both types of images
        cat_path = os.path.join(img_root, category) # the full path to the current category folder (e.g., .../Images/GLAUCOMA)
        if not os.path.exists(cat_path): continue   # checking if the folder exists
        for img_name in os.listdir(cat_path):
            if img_name.lower().endswith((".png", ".jpg")):
                img_id = img_name.split(".")[0]  # dropping the extension
                gt_folder = os.path.join(gt_root, img_id, "SoftMap")    # the path to the folder containing the GT masks
                
                # creating the image pairs
                data_pairs.append({
                    "image": os.path.join(cat_path, img_name),
                    "cup_gt": os.path.join(gt_folder, f"{img_id}_cupsegSoftMap.png"),
                    "disc_gt": os.path.join(gt_folder, f"{img_id}_ODsegSoftMap.png"),
                    "id": img_id    # store the ID for labeling results in the final table
                })
    return data_pairs


# measures how accurate the prediction is
def calculate_metrics(pred_mask, gt_path):
    if not os.path.exists(gt_path): return 0.0, 0.0 # checking if the GT file exists
    gt = cv2.imread(gt_path, 0) # reading the GT image
    if gt is None: return 0.0, 0.0

    # conversion to binary format; don't want any in-between pixel in GTs
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)

    # Dice: ratio of overlap to the total GT area
    intersection = np.sum((pred_bin == 1) & (gt_bin == 1))  # overlap between GT and prediction
    total_gt = np.sum(gt_bin == 1)  # total area in GT that was the Disc/Cup (the actual answer)
    dice = intersection / total_gt if total_gt > 0 else 0.0

    # Accuracy: (Correct pixels) / Total pixels
    correct_pixels = np.sum(pred_bin == gt_bin)
    accuracy = correct_pixels / gt_bin.size

    return dice, accuracy


# image prep work
def run_experiment(samples, disc_perc, cup_perc, save_images=False):
    # lists of performance trackers
    disc_dices, disc_accs = [], []
    cup_dices, cup_accs = [], []

    # naming output folder
    output_dir = "output_samples"
    if save_images: os.makedirs(output_dir, exist_ok=True)

    # processing each pair
    for sample in samples:
        img = cv2.imread(sample['image'])   # fundus image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # conversion to grayscale
        h, w = gray.shape   # dimensions of the fundus image

        # pre-processing: circular mask (to focus on the retina alone)
        center_mask = np.zeros((h, w), dtype=np.uint8)  # darkness (ooh~)
        cv2.circle(center_mask, (w // 2, h // 2), int(min(w, h) * 0.45),
                   255, -1)  # a circle of light (border of the retina)
        masked_gray = cv2.bitwise_and(gray,
                                      center_mask)  # keep the retina; mask out everything outside the circle to darkness

        # task 01: disc (the outer boundary)
        v_disc_min = np.percentile(masked_gray[center_mask > 0],
                                   disc_perc)  # brighter than 90% (parameter) of the pixels
        _, binary_od = cv2.threshold(masked_gray, v_disc_min, 255, cv2.THRESH_BINARY)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_od, connectivity=8)  # 8 connectivity
        # pick the largest white object ("Area" column of the CCA results)
        p_disc = (labels == (1 + np.argmax(stats[1:, 4]))).astype(np.uint8) * 255 if num > 1 else np.zeros_like(
            gray)  # ignore small bright spots in the background

        # task 02: cup (brightest core inside the disc)
        if np.any(p_disc):
            # v set for the cup inside the disc
            v_cup_min = np.percentile(gray[p_disc > 0], cup_perc)  # 75% of the disc we found
            p_cup = cv2.bitwise_and(cv2.inRange(gray, v_cup_min, 255), p_disc)  # bitwise_and to keep it in the disc
        else:
            p_cup = np.zeros_like(gray) # no disc, no cup

        # computing performance metrics
        d_dice, d_acc = calculate_metrics(p_disc, sample['disc_gt'])
        c_dice, c_acc = calculate_metrics(p_cup, sample['cup_gt'])

        # storing for final averaging
        disc_dices.append(d_dice)
        disc_accs.append(d_acc)
        cup_dices.append(c_dice)
        cup_accs.append(c_acc)

        # saving images as output
        if save_images:
            # create a 4-panel comparison to show both disc and cup results
            plt.figure(figsize=(16, 4))

            plt.subplot(1, 4, 1);
            plt.title("Original Fundus");
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 4, 2);
            plt.title("GT (Disc & Cup)");
            plt.imshow(cv2.imread(sample['disc_gt'], 0), cmap='gray')

            plt.subplot(1, 4, 3);
            plt.title("My Pred: Disc");
            plt.imshow(p_disc, cmap='gray')

            plt.subplot(1, 4, 4);
            plt.title("My Pred: Cup");
            plt.imshow(p_cup, cmap='gray')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{sample['id']}_comparison.png")

            plt.close()

    # returning the average scores
    return np.mean(disc_dices), np.mean(disc_accs), np.mean(cup_dices), np.mean(cup_accs)


# main function(sort of?)
dataset_root = "Drishti-GS"  # base directory
test_samples = get_drishti_paths(dataset_root, mode="Testing")

# list as record of my very "scientific" testing
experiments = [
    {"Name": "Experiment 1: Strict", "D%": 98, "C%": 95},   #top 2% bright pixels for disc, top 5% bright pixels (of disc) for cup
    {"Name": "Experiment 2: Moderate", "D%": 95, "C%": 85},   #top 5% bright pixels for disc, top 15% bright pixels (of disc) for cup
    {"Name": "Experiment 3: Optimized", "D%": 90, "C%": 75}   #top 10% bright pixels for disc, top 25% bright pixels (of disc) for cup
]

final_results = []
for exp in experiments:
    # only save the visual plots for the "Optimized" experiment
    is_best = (exp["Name"] == "Experiment 3: Optimized")
    avg_d_dice, avg_d_acc, avg_c_dice, avg_c_acc = run_experiment(test_samples, exp["D%"], exp["C%"],
                                                                  save_images=is_best)
    # organizing the final averages (new dictionary pairs) into a list
    final_results.append({
        "Experiment": exp["Name"],
        "Disc Dice": round(avg_d_dice, 4),
        "Disc Acc": round(avg_d_acc, 4),
        "Cup Dice": round(avg_c_dice, 4),
        "Cup Acc": round(avg_c_acc, 4)
    })

print("\nFinal Results Table (Task 03)\n")
# converting list into a dataframe for tabular alignment
print(pd.DataFrame(final_results).to_string(index=False))