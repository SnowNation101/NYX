# Generating Mixed-Modal Data
This is a guide for generating mixed-model data. We make this a separate part from the main code for a cleaner repo. 


```bash
# if you are at the root dir
cd download_obelics
chmod +x run_download.sh
./run_download.sh
```

check and process the data using process_data.ipynb
run all the cells, the truncted data will be saved to `obelics_chunked_dataset.json`

```bash
python3 data_generation/generate_QAD.py
```
download_data.py: for downloading original obelics dataset
run_download.sh: loop running load_data.py to download dataset
download_images.py: download obelics images
process_data.ipynb: process the downloaded obelics dataset and images
generate_QAD.py: use processed obelics docs to generate (question, doc, answer) data -> generated_qa.json
filter_again.py: filter out data with error loading images -> processed_qa.json
process_generated.py: split generated QAs into single QA pairs -> qa_flattened.json
mine_hardneg.py: use mmE5 to retrieve docs as hard negatives -> obelics_hardneg.json
final_process.py: replace the image paddings, and add image dir to each image name to align with mmE5 datasets -> mm_dataset.json







