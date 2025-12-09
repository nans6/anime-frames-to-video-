
# Attack on GenAI:

**Project Team:**  
Jet Wu ([jetw@andrew.cmu.edu](mailto:jetw@andrew.cmu.edu)), Ananya Sharma ([ananyash@andrew.cmu.edu](mailto:ananyash@andrew.cmu.edu)), Rithwick Sethi ([rithwics@andrew.cmu.edu](mailto:rithwics@andrew.cmu.edu))

---

### Results

- **LoRa Weights:** [attack-on-genai/wan-finetune](https://huggingface.co/attack-on-genai/wan-finetune)
- **Data:** [attack-on-genai/video-frames](https://huggingface.co/datasets/attack-on-genai/video-frames)

---

### Usage

**Step 1:** Download or create a `.zip` file containing your dataset clips and frames, organized as shown in the project structure.

**Step 2:** Upload the `.zip` file to your working environment (e.g., Google Drive if using Colab), then run the provided Colab script to automatically extract the dataset, set up dependencies, and kick off data processing and model finetuning.

**Step 3:** Follow the step-by-step instructions in the Colab notebook to process data, train LoRA weights, and evaluate results.

**Tips:** 

1. Make sure to evaluate on 10+ frames. Otherwise our evaluation metrics won't work. 
2. num_frames must be a multiple of 4k+1, otherwise it won't work.
3. Refresh Colab file manager to see output video after inference step

---

### Acknowledgements

- Video quality metrics codebase: [JunyaoHu/common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality)

