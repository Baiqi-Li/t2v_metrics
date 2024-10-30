import os
import t2v_metrics
from dataset import GenAIBench_Image 
import json
import torch
import numpy as np

root_dir = "./datasets"
result_dir = "./genai_image_results"
output_dir = "./genai_image_results/perSkill"

# models = ["clip-flant5-xxl", "llava-v1.5-13b", "llava-v1.6-13b", "gpt-4o", "openai:ViT-L-14-336", "blip2-itm", "hpsv2", "pickscore-v1", "image-reward-v1"]

# models = ["blip-vqa"]
# models = ["clip-flant5-xxl", "blip-vqa", "llava-v1.5-13b",  "gpt-4o", "instructblip-flant5-xxl", "openai:ViT-L-14-336", "blip2-itm", "image-reward-v1", "pickscore-v1", "hpsv2"]

models = ["clip-flant5-xxl"]

dataset = GenAIBench_Image(root_dir=root_dir)

for model in models:
    result_file = f"{result_dir}/{model}_1600_prompts.pt"
    scores = torch.load(result_file)
    print("Evaluating scores of each skill for model:", model)
    skill_result = dataset.evaluate_scores_per_skill(scores)
    print("Finished evaluating scores for model:", model)
    print("Saving results to:", f"{output_dir}/{model}_1600.json")
    output_file = f"{output_dir}/{model}_1600.json"
    with open(output_file, 'w') as f:
        json.dump(skill_result, f)
    print("Results saved to:", output_file)
    print("\n")
print("\n")    
print("Done!")




# (t2i) [baiqil@trinity-2-29 t2v_metrics]$ python eval_all_skills.py
# Loaded dataset: genai_image.json with 800 prompts

# Evaluating scores of each skill for model: davidsonian
# Pearson's Correlation (no grouping):  37.27165659758984
# Kendall Tau-B Score (no grouping):  0.31918858862448357
# Pairwise Accuracy Score (no grouping):  (0.45801034937834273, 0.005681818181818232)

# Evaluating scores of each skill for model: vq2
# Pearson's Correlation (no grouping):  21.906251835864435
# Kendall Tau-B Score (no grouping):  0.16529059825204634
# Pairwise Accuracy Score (no grouping):  (0.5328487705772036, 0.0)

# models =["davidsonian", "vq2"]

# dataset = GenAI_Bench_Image_800()

# for model in models:
#     if model == "davidsonian":
#         result_file = "/data3/zhiqiul/gpt_score/results/QGA_GenAI_Bench_Image_800_qa_davidsonian_llava-v1.5-13b_final.pth"
#     else:
#         result_file = "/data3/zhiqiul/gpt_score/results/QGA_GenAI_Bench_Image_800_qa_vq2_llava-v1.5-13b_final.pth"
#     scores = torch.load(result_file)
#     print("Evaluating scores of each skill for model:", model)
#     skill_result = dataset.evaluate_scores(scores)

# for model in models:
#     if model == "davidsonian":
#         result_file = "/data3/zhiqiul/gpt_score/results/QGA_GenAI_Bench_Image_800_qa_davidsonian_llava-v1.5-13b_final.pth"
#     else:
#         result_file = "/data3/zhiqiul/gpt_score/results/QGA_GenAI_Bench_Image_800_qa_vq2_llava-v1.5-13b_final.pth"
#     scores = torch.load(result_file)
#     print("Evaluating scores of each skill for model:", model)
#     skill_result = dataset.evaluate_scores_per_skill(scores)
#     print("Finished evaluating scores for model:", model)
#     print("Saving results to:", f"./genai_image_results/perSkill/{model}_800.json")
#     output_file = f"./genai_image_results/perSkill/{model}_800.json"
#     with open(output_file, 'w') as f:
#         json.dump(skill_result, f)
#     print("Results saved to:", output_file)
#     print("\n")
 
# print("Done!")




# 800
#image-reward-v1 :
# Overall Alignment Performance
# Pearson's Correlation (no grouping):  36.177301550167904
# Kendall Tau-B Score (no grouping):  0.25147223227955773
# Pairwise Accuracy Score (no grouping):  (0.5739137493922345, 2.980232238769531e-07)

